import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from flask import Flask, render_template, request, jsonify
from train_models import train_all, load_clean_data, FEATURES, FEATURE_LABELS
from charts import generate_all_charts
import numpy as np
import pickle

app = Flask(__name__)

DATA_PATH   = 'cleaned_student_habits.csv'
MODELS_PATH = 'models/trained.pkl'

def get_results_and_models(force_retrain=False):
    if not force_retrain and os.path.exists(MODELS_PATH):
        print("Loading cached models...")
        with open(MODELS_PATH, 'rb') as f:
            cache = pickle.load(f)
        return cache['results'], cache['trained']
    print("Training all 5 models...")
    results, trained, df = train_all(DATA_PATH)
    os.makedirs('models', exist_ok=True)
    with open(MODELS_PATH, 'wb') as f:
        pickle.dump({'results': results, 'trained': trained}, f)
    generate_all_charts(results, df, static='static')
    return results, trained

import argparse
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--retrain', action='store_true')
_args, _ = _parser.parse_known_args()

RESULTS, TRAINED = get_results_and_models(force_retrain=_args.retrain)
print("Done. Starting Flask...")

MODEL_NAMES = list(RESULTS.keys())
MODEL_KEYS  = ['baseline', 'linear', 'rf', 'gb', 'et', 'svr']
MODEL_MAP   = dict(zip(MODEL_NAMES, MODEL_KEYS))

@app.route('/')
def dashboard():
    df_dash = load_clean_data(DATA_PATH)
    generate_all_charts(RESULTS, df_dash, static='static')
    n_responses = len(df_dash)
    best = max(RESULTS, key=lambda m: RESULTS[m]['r2'] if RESULTS[m]['r2'] > 0 else -999)
    return render_template('index.html',
                           results=RESULTS,
                           best_model=best,
                           model_names=MODEL_NAMES,
                           n_responses=n_responses)

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html', model_names=MODEL_NAMES)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400
    required = ['age','gender','study_hours','attendance','social_media','sleep_hours','diet','exercise','societies']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {chr(44).join(missing)}'}), 400
    model_name = data.get('model', MODEL_NAMES[0])
    key = MODEL_MAP.get(model_name, 'linear')

    entry  = TRAINED[key]
    model  = entry['model']
    scaler = entry['scaler']

    exercise_map = {'0 days': 0, '1-2 days': 1.5, '3-4 days': 3.5, '5-6 days': 5.5, '7 days': 7}
    diet_map     = {'Very Poor': 0, 'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}
    gender_map   = {'Male': 0, 'Female': 1, 'Prefer not to say': 2}

    try:
        # attendance sent as 0-100 from slider — use directly
        row = [[
            float(data['age']),
            gender_map.get(data['gender'], 0),
            float(data['study_hours']),
            float(data['attendance']),      # already 0-100
            float(data['social_media']),
            float(data['sleep_hours']),
            diet_map.get(data['diet'], 2),
            exercise_map.get(data['exercise'], 1.5),
            int(data['societies'])
        ]]
        if scaler:
            row = scaler.transform(row)
        pred = float(np.clip(model.predict(row)[0], 2.0, 4.0))

        if pred >= 3.7:   tier, color = 'Outstanding',        '#39D353'
        elif pred >= 3.3: tier, color = 'Strong Performance', '#00E5FF'
        elif pred >= 2.7: tier, color = 'Average Performance','#F97316'
        else:             tier, color = 'Needs Improvement',  '#EF4444'

        return jsonify({'cgpa': round(pred, 2), 'tier': tier, 'color': color})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
