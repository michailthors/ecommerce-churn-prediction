import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
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

best_params = joblib.load('models/saved/best_params.pkl')

# MLflow experiment
mlflow.set_experiment("ecommerce-conversion-prediction")

with mlflow.start_run():
    # Log params
    mlflow.log_params(best_params)
    mlflow.log_param("scale_pos_weight", scale)

    # Train
    model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Threshold tuning
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    y_pred = (y_prob >= best_threshold).astype(int)

    # Log metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("precision", report['1']['precision'])
    mlflow.log_metric("recall", report['1']['recall'])
    mlflow.log_metric("f1", report['1']['f1-score'])
    mlflow.log_metric("best_threshold", best_threshold)

    # Log model
    mlflow.xgboost.log_model(model, "model")

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall: {report['1']['recall']:.4f}")
    print(f"F1: {report['1']['f1-score']:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print("\nExperiment logged in MLflow!")