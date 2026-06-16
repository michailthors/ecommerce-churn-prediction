import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# Load features
features = pd.read_csv('csv/features.csv')

X = features.drop(columns=['visitorid', 'itemid', 'purchased'])
y = features['purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale = negative / positive

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': scale,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc'
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)

# Τρέξε 20 trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("\n=== BEST PARAMS ===")
print(study.best_params)
print(f"Best ROC-AUC: {study.best_value:.4f}")

# Αποθήκευση best params
joblib.dump(study.best_params, 'models/saved/best_params.pkl')
print("\nBest params saved!")


################################################################################

