# =============================================================================
# CS106: Basics of Machine Learning & Applications
# FINAL PROJECT FILE (FIXED & STABLE v3)
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, classification_report, ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# LOAD SURVEY DATA
# =============================================================================
df_survey = pd.read_csv("cleaned_form.csv")

df_survey = df_survey.rename(columns={
    "Age": "age_group",
    "Occupation": "occupation",
    "Average Daily Screentime (hours": "daily_screentime",
    "Social Media usage (per day [hours])": "social_media_hours",
    "How many times do you check your phone per day? (pickups)": "phone_pickups",
    "Do you use your phone right before sleeping? (30mins before)": "phone_before_sleep",
    "Average Sleep Duration (hours)": "sleep_hours",
    "Do you exercise regularly? (3+ days per week)": "exercise_freq",
    "Study/Work Hours per week": "study_work_hours",
    "How productive do you consider yourself": "productivity_score"
})

df_survey["source"] = "survey"

# =============================================================================
# LOAD KAGGLE DATA
# =============================================================================
try:
    df_kaggle = pd.read_csv("kaggle_screentime.csv")

    df_kaggle = df_kaggle.rename(columns={
        "age": "age_group",
        "daily_screen_time_hours": "daily_screentime",
        "phone_usage_before_sleep_minutes": "phone_before_sleep",
        "sleep_duration_hours": "sleep_hours",
        "physical_activity_minutes": "exercise_freq",
        "notifications_received_per_day": "phone_pickups"
    })

    df_kaggle["productivity_score"] = 10 - df_kaggle["mental_fatigue_score"]
    df_kaggle["source"] = "kaggle"

    df = pd.concat([df_survey, df_kaggle], ignore_index=True, sort=False)
    print("Loaded both survey and Kaggle data.")

except Exception as e:
    df = df_survey.copy()
    print(f"Kaggle data not found, using survey only. ({e})")

print(f"Total rows loaded: {len(df)}")

# =============================================================================
# CLEANING
# =============================================================================
categorical_cols = ["age_group", "phone_before_sleep", "exercise_freq", "occupation"]

le = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = le.fit_transform(df[col])

cols_to_numeric = [
    "productivity_score", "daily_screentime", "social_media_hours",
    "phone_pickups", "sleep_hours", "study_work_hours"
]

for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["productivity_score"], inplace=True)

threshold = df["productivity_score"].median()
df["label"] = (df["productivity_score"] >= threshold).astype(int)

all_features = [
    "age_group", "daily_screentime", "social_media_hours",
    "phone_pickups", "phone_before_sleep", "sleep_hours",
    "exercise_freq", "study_work_hours"
]

features = [f for f in all_features if f in df.columns]

X = df[features].copy()
y = df["label"]

# =============================================================================
# SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Imputation
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# SMOTE
if min(pd.Series(y_train).value_counts()) >= 6:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("SMOTE applied.")
else:
    print("SMOTE skipped.")

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================================
# MODELS + GRID SEARCH
# =============================================================================
param_grids = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    }
}

best_models = {}

for name, config in param_grids.items():
    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_

    print(f"\n{name}")
    print("Best Params:", grid.best_params_)

# =============================================================================
# EVALUATION
# =============================================================================
results = []
trained_models = {}

for name, model in best_models.items():
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob)
    }

    results.append(row)
    trained_models[name] = (model, pred, prob)

    print(f"\n{name}")
    print(row)
    print(classification_report(y_test, pred))

results_df = pd.DataFrame(results)
print("\nSummary:\n", results_df)

# =============================================================================
# VISUALIZATION (FIXED)
# =============================================================================
n_models = len(trained_models)

fig = plt.figure(figsize=(6 * n_models, 10))
gs = gridspec.GridSpec(2, n_models)

# --- Bar chart ---
ax1 = fig.add_subplot(gs[0, :])
ax1.bar(results_df["Model"], results_df["F1"])
ax1.set_title("F1 Score Comparison")
ax1.set_xlabel("Models")
ax1.set_ylabel("F1 Score")

# --- Confusion Matrices ---
for i, (name, (model, pred, prob)) in enumerate(trained_models.items()):
    ax = fig.add_subplot(gs[1, i])
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    ax.set_title(name)

plt.tight_layout()
plt.savefig("model_output.png")
plt.show()