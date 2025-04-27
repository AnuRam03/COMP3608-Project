from tabnanny import verbose
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn import clone
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from DataSet3.HelpFunctions.metric_functions import evaluate_classification_model

#figured out that adding the args and stuff actually shows it when u hover over the function, so i add it for better explanation :D
def model_comparison(models_dict, scalers, samplers, X, y, numerical_cols, test_size=0.2, random_state=42, verbose=0):
    """
    Runs multiple models through combinations of scaling and resampling techniques, returning results.

    Args:
        models_dict (dict): Dictionary where keys are model names (str) and
                            values are instantiated scikit-learn model objects.
        scalers (dcit): Dictionary where keys are types of scaler names (str) and values
                            are instantiated scalers
        sampler (dcit): Dictionary where keys are types of resampling technique names (str)
                            and the values are instantied Samplers
        X (pd.DataFrame or np.ndarray): Feature data.
        y (pd.Series or np.ndarray): Target data.
        numerical_cols (list): List of column names in X to apply scaling to.
                                Should only contain columns present in X.
        test_size (float, optional): Proportion of dataset for the test split. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for all combinations.
                      Sorted by ROC AUC and F1 Score (descending).
    """

    # --- Input Validation ---
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("X must be a pandas DataFrame or numpy array.")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y must be a pandas Series or numpy array.")
    if isinstance(X, pd.DataFrame):
        missing_num_cols = [col for col in numerical_cols if col not in X.columns]
        if missing_num_cols:
            raise ValueError(f"Numerical columns not found in X: {missing_num_cols}")
    elif isinstance(X, np.ndarray):
         print("Warning: X is a NumPy array. Assuming numerical_cols indices are correct if scaling is used.")
         # Basic check if indices are valid
         if numerical_cols and (max(numerical_cols) >= X.shape[1] or min(numerical_cols) < 0):
              raise ValueError("numerical_cols contains invalid indices for the NumPy array X.")


    # --- Train/Test Split ---
    if (verbose > 0): print(f"Splitting data: test_size={test_size}, random_state={random_state}, stratify=y")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if (verbose > 0):
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"Test label distribution:\n{pd.Series(y_test).value_counts(normalize=True)}\n")

    # --- Results Storage ---
    results_list = []

    # --- Looping through combinations ---
    for model_name, model_base_instance in models_dict.items():
        if (verbose > 0): print(f"\n=== Evaluating Model: {model_name} ===")
        for scaler_name, scaler_instance in scalers.items():
            if (verbose > 2): print(f"  --- Scaler: {scaler_name} ---")

            # --- Apply Scaling ---
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()
            current_scaler = None # To hold the fitted scaler

            if scaler_instance:
                if (verbose > 2): print(f"    Fitting and transforming with {scaler_name}...")
                current_scaler = clone(scaler_instance) # Clone scaler for fresh fit
                try:
                    if isinstance(X_train, pd.DataFrame):
                        # Fit only on numerical columns of the training data
                        current_scaler.fit(X_train_processed[numerical_cols])
                        X_train_processed[numerical_cols] = current_scaler.transform(X_train_processed[numerical_cols])
                        X_test_processed[numerical_cols] = current_scaler.transform(X_test_processed[numerical_cols])
                    else: # Handle NumPy array case
                         # Fit and transform numerical columns specified by indices
                         current_scaler.fit(X_train_processed[:, numerical_cols])
                         X_train_processed[:, numerical_cols] = current_scaler.transform(X_train_processed[:, numerical_cols])
                         X_test_processed[:, numerical_cols] = current_scaler.transform(X_test_processed[:, numerical_cols])

                except Exception as e:
                    print(f"    ERROR applying scaler {scaler_name}: {e}")
                    continue

            else:
                if (verbose > 2): print("    No scaling applied.")

            for sampler_name, sampler_instance in samplers.items():
                if (verbose > 2): print(f"    --- Sampler: {sampler_name} ---")

                # --- Apply Sampling (Train Only) ---
                X_train_final = X_train_processed.copy()
                y_train_final = y_train.copy()
                current_sampler = None # To hold the fitted sampler

                if sampler_instance:
                    if (verbose > 2): print(f"      Applying {sampler_name} to training data...")
                    current_sampler = clone(sampler_instance) # Clone sampler
                    try:
                        X_train_final, y_train_final = current_sampler.fit_resample(X_train_final, y_train_final)
                        if (verbose > 2): print(f"      Training data shape after {sampler_name}: {X_train_final.shape}")
                    except Exception as e:
                        print(f"      ERROR applying {sampler_name}: {e}")
                        continue 
                else:
                    if (verbose > 2): print("      No sampling applied to training data.")

                # --- Train ---
                if (verbose > 2): print(f"        Training {model_name}...")
                current_model = clone(model_base_instance) # Clone model for a fresh instance
                try:
                    current_model.fit(X_train_final, y_train_final)
                except Exception as e:
                     print(f"        ERROR training {model_name} with {scaler_name}/{sampler_name}: {e}")
                     continue

                # --- Evaluate ---
                if (verbose > 2): print(f"        Evaluating {model_name} on test data...")
                try:
                    # Pass processed TEST data (X_test_processed)
                    metrics = evaluate_classification_model(
                        current_model, model_name, X_test_processed, y_test, scaler_name, sampler_name
                    )
                    results_list.append(metrics)
                except Exception as e:
                     print(f"        ERROR evaluating {model_name} with {scaler_name}/{sampler_name}: {e}")
                     continue # Skip if evaluation fails

    # --- Consolidate and Return ---
    if not results_list:
        print("No results were generated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)
    # Sort by performance 
    results_df = results_df.sort_values(
        by=['F1 Score','ROC AUC'], ascending=False, na_position='last'
    ).reset_index(drop=True)

    print("\n=== Pipeline Execution Complete ===")
    return results_df
