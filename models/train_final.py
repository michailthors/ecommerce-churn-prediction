import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

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

# Φόρτωσε τα best params από το Optuna
best_params = joblib.load('models/saved/best_params.pkl')
print(f"Loaded best params: {best_params}")

# Train τελικό model
final_model = XGBClassifier(
    **best_params,
    scale_pos_weight=scale,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc'
)

final_model.fit(X_train, y_train)

y_prob_final = final_model.predict_proba(X_test)[:, 1]

# Threshold tuning
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_final)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

y_pred_final = (y_prob_final >= best_threshold).astype(int)

print("\n=== FINAL MODEL ===")
print(f"Best threshold: {best_threshold:.4f}")
print(classification_report(y_test, y_pred_final))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_final):.4f}")

# Αποθήκευση
joblib.dump(final_model, 'models/saved/final_model.pkl')
joblib.dump(best_threshold, 'models/saved/best_threshold.pkl')
print("\nFinal model saved!")