# Habits to Grades

> **Can student lifestyle habits predict academic performance?**  
> A full-stack machine learning project built on a survey (augmented) of 653 GIKI students.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-black?style=flat&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?style=flat&logo=pandas)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

---

## Overview

This project trains and compares **5 regression models** to predict a student's CGPA from 9 behavioral features — study hours, attendance, sleep, social media usage, diet, exercise, and more.

It ships as an interactive Flask dashboard with:
- A **model comparison table** (MAE, RMSE, Test R², CV R²)
- Auto-generated **feature importance**, **correlation heatmap**, and **CV bar** charts
- A **live CGPA predictor** — fill in your habits, get an instant prediction from any of the 5 models

---

## Dataset

| Attribute        | Value                                          |
|------------------|------------------------------------------------|
| Source           | Google Forms survey — GIKI students only       |
| Responses        | **653** (after cleaning: varies by model run)  |
| Gender split     | Male 80% · Female 16% · Other 4%              |
| CGPA range       | 2.0 – 4.0 (mean **2.83**, std **0.35**)        |
| Collection period| April – May 2026                               |

**Features used for modeling:**

| Feature         | Type        | Notes                                       |
|-----------------|-------------|---------------------------------------------|
| Age             | Numeric     | —                                           |
| Gender          | Categorical | Encoded: Male=0, Female=1, Other=2          |
| Study Hours     | Numeric     | Hours/day — parsed from free-text responses |
| Attendance %    | Numeric     | 0–100                                       |
| Social Media    | Numeric     | Hours/day                                   |
| Sleep Hours     | Numeric     | Hours/night                                 |
| Diet Quality    | Ordinal     | Very Poor → Excellent (0–4)                 |
| Exercise        | Ordinal     | Days/week — mapped to numeric midpoints     |
| Societies       | Numeric     | Count of clubs/societies joined             |

---

## Models

| Model              | Key Hyperparameters                              | Scaling |
|--------------------|--------------------------------------------------|---------|
| Linear Regression  | Default                                          | None    |
| Random Forest      | n_estimators=200, max_depth=6, min_samples_leaf=3 | None   |
| Gradient Boosting  | n_estimators=200, max_depth=3, lr=0.05           | None    |
| Extra Trees        | n_estimators=200, max_depth=6, min_samples_leaf=3 | None   |
| SVR                | kernel=rbf, C=10, ε=0.05, gamma=scale            | StandardScaler |

All models are evaluated with **5-fold cross-validation** on CV R² and compared on held-out test MAE/RMSE/R².

---

## Key Findings

- **Attendance** is the strongest single predictor of CGPA.
- **Study hours** shows a clear positive linear trend.
- **Social media usage** is negatively correlated — more time on social media correlates with lower CGPA.
- **Diet and exercise** show weak but directionally positive correlations.
- Random Forest achieved the strongest held-out Test R², while Linear Regression achieved the highest 5-fold CV R². 
- This suggests the dataset has meaningful linear patterns, while tree-based models may capture extra non-linear relationships on the test split.

---

## Project Structure

```
habits_to_grades/
├── app.py                  # Flask app — routes, prediction API, model caching
├── data.xlsx               # Raw survey responses
├── requirements.txt        # Pinned dependencies
├── models/
│   ├── train_models.py     # Data cleaning, feature engineering, model training
│   └── charts.py           # Matplotlib/Seaborn chart generation
├── templates/
│   ├── base.html           # Shared layout (dark theme)
│   ├── index.html          # Dashboard page
│   └── predict.html        # Live predictor page
├── static/
│   └── plots/              # Generated charts (auto-created on startup)
└── notebooks/
    └── eda.ipynb           # Exploratory data analysis
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/A-1K/tds-proj-student-cgpa-predictor.git
cd habits-to-grades

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
# → http://127.0.0.1:5000
```

On first run, all 5 models are trained and cached to `models/trained.pkl`.  
Subsequent starts load the cache instantly. To force a retrain:

```bash
python app.py --retrain
```

---

## API

The predictor is also available as a JSON endpoint:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gradient Boosting",
    "age": 20,
    "gender": "Male",
    "study_hours": 3.5,
    "attendance": 85,
    "social_media": 2.0,
    "sleep_hours": 7.0,
    "diet": "Average",
    "exercise": "3-4 days",
    "societies": 1
  }'
```

**Response:**
```json
{
  "cgpa": 3.12,
  "tier": "Strong Performance",
  "color": "#00E5FF"
}
```

**Valid values:**
- `model`: `"Linear Regression"`, `"Random Forest"`, `"Gradient Boosting"`, `"Extra Trees"`, `"SVR"`
- `gender`: `"Male"`, `"Female"`, `"Prefer not to say"`
- `diet`: `"Very Poor"`, `"Poor"`, `"Average"`, `"Good"`, `"Excellent"`
- `exercise`: `"0 days"`, `"1-2 days"`, `"3-4 days"`, `"5-6 days"`, `"7 days"`

---

## Course Context

Built as the final sem project for **DS-211 Theory of Data Science** at [GIK Institute of Engineering Sciences and Technology](https://www.giki.edu.pk/).

---

## License

[MIT](LICENSE)
