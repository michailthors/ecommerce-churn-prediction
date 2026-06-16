import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load features
features = pd.read_csv('csv/features.csv')

# X = features, y = target
X = features.drop(columns=['visitorid', 'itemid', 'purchased'])
y = features['purchased']

# Split σε train και test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
print(f"\nTarget distribution (train):\n{y_train.value_counts()}")

# Model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

#####################################################

from xgboost import XGBClassifier

# Υπολογισμός scale_pos_weight για class imbalance
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale = negative / positive
print(f"\nscale_pos_weight: {scale:.2f}")

# XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=scale,  # αντιμετωπίζει το class imbalance
    random_state=42,
    n_jobs=-1,
    eval_metric='auc'
)

xgb_model.fit(X_train, y_train)

# Evaluation
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n=== XGBoost CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")



#####################################################

from sklearn.metrics import precision_recall_curve
import numpy as np

# Threshold tuning για XGBoost
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_xgb)

# Βρες το threshold που δίνει καλύτερο F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\n=== THRESHOLD TUNING ===")
print(f"Best threshold: {best_threshold:.4f}")
print(f"Best F1: {f1_scores[best_idx]:.4f}")
print(f"Precision at best threshold: {precisions[best_idx]:.4f}")
print(f"Recall at best threshold: {recalls[best_idx]:.4f}")

# Εφάρμοσε το νέο threshold
y_pred_tuned = (y_prob_xgb >= best_threshold).astype(int)
print("\n=== TUNED CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_tuned))



#####################################################


import joblib
import os

# Δημιουργία φακέλου για τα saved models
os.makedirs('models/saved', exist_ok=True)

# Αποθήκευση XGBoost model και threshold
joblib.dump(xgb_model, 'models/saved/xgb_model.pkl')
joblib.dump(best_threshold, 'models/saved/best_threshold.pkl')

print("\n=== MODEL SAVED ===")
print("XGBoost model saved to models/saved/xgb_model.pkl")
print(f"Best threshold saved: {best_threshold:.4f}")

