# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODELS = {
    "ElasticNet": LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        random_state=42,
        n_estimators=100
    ),
    "CatBoost": CatBoostClassifier(
        random_seed=42,
        verbose=0,
        iterations=100
    )
}

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    """
    Trains a model and evaluates its performance.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : array-like
        Training and testing data.
    model_name : str
        Name of the model to use (must be a key in the MODELS dictionary).

    Returns
    -------
    tuple
        A tuple containing the trained model and a dictionary with the metrics.
    """
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not recognized. Available models: {list(MODELS.keys())}")

    model = MODELS[model_name]

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }

    return model, metrics

import joblib
from pathlib import Path

def save_model(model, full_save_path):
    """
    Saves a trained model to the specified full path.

    Parameters
    ----------
    model : estimator
        The trained model object to save.
    full_save_path : str or pathlib.Path
        The full, explicit path (including filename) where the model should be saved.
    """
    # Ensure the target directory exists
    save_path = Path(full_save_path)
    save_dir = save_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, save_path)
    print(f"Model saved successfully to {save_path}")
