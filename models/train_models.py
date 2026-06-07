import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
def parse_num(v):
    v = str(v).strip().lower()
    m = re.match(r'^(\d+\.?\d*)\s*(hrs?|hours?|h)?$', v)
    if m: return float(m.group(1))
    m = re.match(r'^(\d+\.?\d*)\s*[-\/]+\s*(\d+\.?\d*)', v)
    if m: return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.match(r'^(\d+\.?\d*)\s+to\s+(\d+\.?\d*)', v)
    if m: return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.match(r'^(\d+\.?\d*)\\+', v)
    if m: return float(m.group(1))
    try: return float(v)
    except: return np.nan

def load_clean_data(path='cleaned_student_habits.csv'):
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, engine='openpyxl')

    df.columns = [str(col).strip() for col in df.columns]
    if len(df.columns) == 13:
        df.columns = ['timestamp','name','age','gender','study_hours','attendance',
                      'cgpa','sgpa','social_media','sleep_hours','diet_quality','exercise','societies']
    elif len(df.columns) == 11:
        df.columns = ['age','gender','study_hours','attendance',
                      'cgpa','sgpa','social_media','sleep_hours','diet_quality','exercise','societies']
    else:
        raise ValueError(f'Unexpected dataset shape: expected 11 or 13 columns, got {len(df.columns)}')

    exercise_map = {'0 days': 0, '1-2 days': 1.5, '3-4 days': 3.5, '5-6 days': 5.5, '7 days': 7}
    diet_map = {
        'Very Poor (Highly reliant on junk food and caffeine)': 0,
        'Poor (Mostly processed foods, soft drinks, limited vegetables)': 1,
        'Average (Mix of healthy and unhealthy)': 2,
        'Good (Mostly balanced, some fast food/unhealthy snacks)': 3,
        'Excellent (Well-balanced, fresh food)': 4
    }
    gender_map = {'Male': 0, 'Female': 1, 'Prefer not to say': 2}

    df['study_hours']    = df['study_hours'].apply(parse_num)
    df['social_media']   = df['social_media'].apply(parse_num)
    df['cgpa']           = pd.to_numeric(df['cgpa'], errors='coerce')
    df['age']            = pd.to_numeric(df['age'],  errors='coerce')
    df['exercise_num']   = df['exercise'].map(exercise_map)
    df['attendance_pct'] = pd.to_numeric(df['attendance'], errors='coerce')  # already 0-100
    df['diet_enc']       = df['diet_quality'].map(diet_map)
    df['gender_enc']     = df['gender'].map(gender_map)

    def count_soc(v):
        if pd.isna(v): return 0
        v = str(v).strip().lower()
        if v in ['n/a','na','none','no','nil','-','nan','']: return 0
        return len([x for x in v.split(',') if x.strip()])

    df['society_count'] = df['societies'].apply(count_soc)

    df_clean = df.dropna(subset=[
        'cgpa','age','study_hours','social_media',
        'diet_enc','gender_enc','exercise_num','attendance_pct',
        'sleep_hours'
    ]).copy()

    return df_clean

FEATURES = ['age','gender_enc','study_hours','attendance_pct',
            'social_media','sleep_hours','diet_enc','exercise_num','society_count']
FEATURE_LABELS = ['Age','Gender','Study Hours','Attendance %',
                  'Social Media','Sleep Hours','Diet Quality','Exercise','Societies']

# ─────────────────────────────────────────────
# TRAIN ALL 5 MODELS
# ─────────────────────────────────────────────
def train_all(data_path='cleaned_student_habits.csv'):
    df = load_clean_data(data_path)
    X  = df[FEATURES]
    y  = df['cgpa']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_s       = scaler.transform(X)

    results = {}
    trained = {}

    # ── 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    cv = cross_val_score(lr, X, y, cv=5, scoring='r2')
    results['Linear Regression'] = _metrics(y_test, y_pred, cv)
    results['Linear Regression']['coef'] = dict(zip(FEATURE_LABELS, lr.coef_))
    trained['linear'] = {'model': lr, 'scaler': None}

    # ── 2. Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    cv = cross_val_score(rf, X, y, cv=5, scoring='r2')
    results['Random Forest'] = _metrics(y_test, y_pred, cv)
    results['Random Forest']['importance'] = dict(zip(FEATURE_LABELS, rf.feature_importances_))
    trained['rf'] = {'model': rf, 'scaler': None}

    # ── 3. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    min_samples_leaf=3, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    cv = cross_val_score(gb, X, y, cv=5, scoring='r2')
    results['Gradient Boosting'] = _metrics(y_test, y_pred, cv)
    results['Gradient Boosting']['importance'] = dict(zip(FEATURE_LABELS, gb.feature_importances_))
    trained['gb'] = {'model': gb, 'scaler': None}

    # ── 4. Extra Trees
    et = ExtraTreesRegressor(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42)
    et.fit(X_train, y_train)
    y_pred = et.predict(X_test)
    cv = cross_val_score(et, X, y, cv=5, scoring='r2')
    results['Extra Trees'] = _metrics(y_test, y_pred, cv)
    results['Extra Trees']['importance'] = dict(zip(FEATURE_LABELS, et.feature_importances_))
    trained['et'] = {'model': et, 'scaler': None}

    # ── 5. SVR
    svr = SVR(kernel='rbf', C=10, epsilon=0.05, gamma='scale')
    svr.fit(X_train_s, y_train)
    y_pred = svr.predict(X_test_s)
    cv = cross_val_score(svr, X_s, y, cv=5, scoring='r2')
    results['SVR'] = _metrics(y_test, y_pred, cv)
    trained['svr'] = {'model': svr, 'scaler': scaler}

    trained['scaler'] = scaler

    # Best = highest test R2, excluding any model with negative test R2 if possible
    pos = {m: r for m, r in results.items() if r['r2'] > 0}
    pool = pos if pos else results
    best_model = max(pool, key=lambda m: results[m]['r2'])
    for m in results:
        results[m]['is_best'] = (m == best_model)

    return results, trained, df

def _metrics(y_test, y_pred, cv_scores):
    return {
        'mae':    round(float(mean_absolute_error(y_test, y_pred)), 3),
        'rmse':   round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
        'r2':     round(float(r2_score(y_test, y_pred)), 3),
        'cv_r2':  round(float(cv_scores.mean()), 3),
        'cv_std': round(float(cv_scores.std()), 3),
    }

def save_models(trained, out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'trained.pkl'), 'wb') as f:
        pickle.dump(trained, f)

def load_models(out_dir='models'):
    with open(os.path.join(out_dir, 'trained.pkl'), 'rb') as f:
        return pickle.load(f)
