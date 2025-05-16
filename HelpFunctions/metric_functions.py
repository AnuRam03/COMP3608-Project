import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score,
                           precision_score,
                           recall_score,
                           f1_score,
                           roc_curve,
                           auc,
                           roc_auc_score)
from sklearn.base import is_classifier

def evaluate_classification_model(model, model_name, X_test, y_test, scaler_name = None, sampler_name = None):
    """Evaluates model and returns metrics in a dictionary, including preprocessing info.
    
    Args:
        model: scikit-learn model object
        model_name (str): name of the model (will show in the returne dcit)
        X_test (pd.DataFrame or np.ndarray): Feature data.
        y_test (pd.Series or np.ndarray): Target data.
        scaler_name (str): name of scaler used on the data
        sampler_name (str): name of the resampling method used on the feature data

    Returns:
        dict[str, Any]: Dictionary containing metrics produced by the model
    """
    y_pred = model.predict(X_test)
    roc_auc = float('nan') # Initialize as NaN

    # --- Probability/Score Predictions (for ROC AUC) ---
    y_scores = None
    if hasattr(model, "predict_proba") and callable(model.predict_proba):
        try:
            y_scores = model.predict_proba(X_test)[:, 1]
        except Exception as e:
             print(f"Error getting predict_proba for {model_name}: {e}")
    elif hasattr(model, "decision_function") and callable(model.decision_function):
        try:
            y_scores = model.decision_function(X_test)
            # Ensure y_scores is 1D
            if y_scores.ndim > 1 and y_scores.shape[1] > 1:
                 y_scores = y_scores[:, 1] # Assuming binary, take second column
            elif y_scores.ndim > 1:
                 y_scores = y_scores.ravel() # Flatten if (n, 1)
            print(f"Info: Used decision_function() for ROC AUC ({model_name}).")
        except Exception as e:
             print(f"Error getting decision_function for {model_name}: {e}")
    else:
         print(f"Warning: Model '{model_name}' supports neither predict_proba() nor decision_function(). ROC AUC unavailable.")

    # Calculate ROC AUC if scores were obtained
    if y_scores is not None:
        try:
             roc_auc = roc_auc_score(y_test, y_scores)
        except Exception as e:
             print(f"Error calculating ROC AUC for {model_name}: {e}. ROC AUC set to NaN.")
             roc_auc = float('nan') # Set back to NaN on error

    # --- Calculate Other Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # --- Store Results ---
    metrics = {
        'Model': model_name,
        'Scaler': scaler_name,
        'Sampler': sampler_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }
    return metrics


def plot_roc_curve(model, X_test, y_test, model_name):
    # Check if the model supports probability prediction
    if not (hasattr(model, "predict_proba") and callable(model.predict_proba)):
        print(f"Model '{model_name}' does not support predict_proba(). Cannot plot ROC curve.")
        # You might want to check for decision_function as an alternative for some models like SVC
        if hasattr(model, "decision_function") and callable(model.decision_function):
             print("Trying decision_function() instead...")
             try:
                 y_scores = model.decision_function(X_test)
             except Exception as e:
                 print(f"Could not use decision_function: {e}")
                 return # Exit if neither works
        else:
            return # Exit if neither predict_proba nor decision_function is available

    else:
        # Get predicted probabilities for the positive class (usually class 1)
        y_scores = model.predict_proba(X_test)[:, 1]


    # Calculate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # Calculate the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)
    # Alternatively using roc_auc_score: roc_auc = roc_auc_score(y_test, y_scores)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'{model_name} (AUC = {roc_auc:.3f})') # Format AUC to 3 decimal places
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance') # Diagonal line

    # Formatting the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05]) # Slightly higher limit to see the top border
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()