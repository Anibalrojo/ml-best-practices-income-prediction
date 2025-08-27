import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import os

def plot_model_evaluation(y_true, model, X_test, model_name, save_path_prefix=None):
    """
    Generates, displays, and optionally saves the confusion matrix and ROC curve for a model.

    Parameters
    ----------
    y_true : array-like
        True target values.
    model : estimator
        The trained classification model.
    X_test : array-like
        Test data for prediction.
    model_name : str
        Name of the model for plot titles.
    save_path_prefix : str, optional
        If provided, saves the plots to files. For example, a prefix of
        '../reports/figures/elasticnet' would save 'elasticnet_confusion_matrix.png'
        and 'elasticnet_roc_curve.png' in the specified directory.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    
    if save_path_prefix:
        path = f"{save_path_prefix}_confusion_matrix.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f"Saved confusion matrix to {path}")
    
    plt.show()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc='lower right')
    plt.grid()
    
    if save_path_prefix:
        path = f"{save_path_prefix}_roc_curve.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f"Saved ROC curve to {path}")
        
    plt.show()
