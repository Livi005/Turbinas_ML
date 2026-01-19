"""Model validation and evaluation utilities."""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Handle imports for both script and module execution
try:
    from src.augmentation.smote_augment import apply_smote
except ImportError:
    # If running as module, try relative import
    sys.path.insert(
        0,
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
    from src.augmentation.smote_augment import apply_smote

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Metadata utilities (NEW)
# -----------------------------
def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load a metadata JSON file saved alongside a trained model.

    This is designed to be robust to different key naming conventions used
    in Optuna exports or custom metadata formats.

    Parameters
    ----------
    metadata_path : str
        Path to metadata JSON.

    Returns
    -------
    dict
        Parsed metadata dictionary.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if not isinstance(meta, dict):
        raise ValueError("Metadata JSON must contain a JSON object (dict).")

    return meta


def extract_hyperparams_from_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model hyperparameters from metadata.

    Tries multiple common keys:
    - meta["best_params"]
    - meta["hyperparameters"]
    - meta["params"]
    - meta["model_params"]
    - meta["optuna"]["best_params"]
    - meta["best_trial"]["params"]

    Returns empty dict if not found.

    Parameters
    ----------
    meta : dict
        Metadata dictionary.

    Returns
    -------
    dict
        Hyperparameter dictionary for model construction.
    """
    candidate_paths = [
        ("best_params",),
        ("hyperparameters",),
        ("params",),
        ("model_params",),
        ("optuna", "best_params"),
        ("best_trial", "params"),
    ]

    for path in candidate_paths:
        cur = meta
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and isinstance(cur, dict):
            return cur

    return {}


def extract_training_flags_from_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract optional training/validation flags from metadata.
    These keys are OPTIONAL and only used if present.

    Supported:
    - use_smote
    - apply_scaling
    - random_state
    - cv
    - cv_repeats
    - metrics

    Returns
    -------
    dict
        A dictionary with any found flags.
    """
    flags = {}
    for key in ["use_smote", "apply_scaling", "random_state", "cv", "cv_repeats", "metrics"]:
        if key in meta:
            flags[key] = meta[key]

    # Sometimes these might be nested
    if "training" in meta and isinstance(meta["training"], dict):
        for key in ["use_smote", "apply_scaling", "random_state", "cv", "cv_repeats", "metrics"]:
            if key in meta["training"] and key not in flags:
                flags[key] = meta["training"][key]

    return flags


# -----------------------------
# Core validation utilities
# -----------------------------
def _to_numpy_Xy(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = np.asarray(X)

    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.asarray(y)

    return X_array, y_array


def _default_metric_functions():
    return {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        ),
        'recall': lambda y_true, y_pred: recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        ),
        'f1': lambda y_true, y_pred: f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        ),
        # roc_auc is handled separately because it needs probabilities
        'roc_auc': None
    }


def cross_validate_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 5,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    random_state: int = 42,
    use_smote: bool = True,
    apply_scaling: bool = True
) -> Dict[str, Any]:
    """
    Perform Stratified K-Fold Cross-Validation with SMOTE augmentation applied
    only to training data within each fold.

    Returns a dict with fold_results, mean, std, cv_folds.
    """
    X_array, y_array = _to_numpy_Xy(X, y)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    fold_results = []
    metric_functions = _default_metric_functions()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array)):
        X_train_fold = X_array[train_idx]
        X_val_fold = X_array[val_idx]
        y_train_fold = y_array[train_idx]
        y_val_fold = y_array[val_idx]

        # Apply SMOTE only to training fold
        if use_smote:
            X_train_aug, y_train_aug = apply_smote(
                X_train_fold,
                y_train_fold,
                random_state=random_state
            )
        else:
            X_train_aug = X_train_fold
            y_train_aug = y_train_fold

        # Scaling
        if apply_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_aug)
            X_val_scaled = scaler.transform(X_val_fold)
        else:
            X_train_scaled = X_train_aug
            X_val_scaled = X_val_fold

        # Train fresh copy of model
        model_copy = type(model)(**model.get_params())
        model_copy.fit(X_train_scaled, y_train_aug)

        y_pred_fold = model_copy.predict(X_val_scaled)

        fold_metrics = {}
        for metric_name in metrics:
            if metric_name == 'roc_auc':
                try:
                    y_pred_proba = model_copy.predict_proba(X_val_scaled)[:, 1]
                    fold_metrics[metric_name] = roc_auc_score(y_val_fold, y_pred_proba)
                except (AttributeError, IndexError, ValueError):
                    fold_metrics[metric_name] = 0.0
            else:
                fn = metric_functions.get(metric_name)
                fold_metrics[metric_name] = fn(y_val_fold, y_pred_fold) if fn else 0.0

        fold_metrics['fold'] = fold_idx + 1
        fold_results.append(fold_metrics)

    mean_metrics = {}
    std_metrics = {}
    for metric_name in metrics:
        metric_values = [fold.get(metric_name, 0.0) for fold in fold_results]
        mean_metrics[metric_name] = float(np.mean(metric_values)) if metric_values else 0.0
        std_metrics[metric_name] = float(np.std(metric_values)) if metric_values else 0.0

    return {
        'fold_results': fold_results,
        'mean': mean_metrics,
        'std': std_metrics,
        'cv_folds': cv
    }


def cross_validate_model_repeated(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 10,
    repeats: int = 3,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    random_state: int = 42,
    use_smote: bool = True,
    apply_scaling: bool = True
) -> Dict[str, Any]:
    """
    Repeated Stratified K-Fold validation:
    - Runs StratifiedKFold(cv) 'repeats' times with different shuffles/seeds.
    - Returns fold-level results of size (cv * repeats), plus mean/std.

    This matches the idea: K10 with 3 repeats -> 30 scores per model.
    """
    X_array, y_array = _to_numpy_Xy(X, y)
    metric_functions = _default_metric_functions()

    all_fold_results: List[Dict[str, Any]] = []

    for r in range(repeats):
        seed = random_state + r
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array), start=1):
            X_train_fold = X_array[train_idx]
            X_val_fold = X_array[val_idx]
            y_train_fold = y_array[train_idx]
            y_val_fold = y_array[val_idx]

            # SMOTE only on training fold
            if use_smote:
                X_train_aug, y_train_aug = apply_smote(
                    X_train_fold,
                    y_train_fold,
                    random_state=seed
                )
            else:
                X_train_aug = X_train_fold
                y_train_aug = y_train_fold

            # Scaling
            if apply_scaling:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_aug)
                X_val_scaled = scaler.transform(X_val_fold)
            else:
                X_train_scaled = X_train_aug
                X_val_scaled = X_val_fold

            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train_scaled, y_train_aug)

            y_pred_fold = model_copy.predict(X_val_scaled)

            fold_metrics = {}
            for metric_name in metrics:
                if metric_name == 'roc_auc':
                    try:
                        y_pred_proba = model_copy.predict_proba(X_val_scaled)[:, 1]
                        fold_metrics[metric_name] = roc_auc_score(y_val_fold, y_pred_proba)
                    except (AttributeError, IndexError, ValueError):
                        fold_metrics[metric_name] = 0.0
                else:
                    fn = metric_functions.get(metric_name)
                    fold_metrics[metric_name] = fn(y_val_fold, y_pred_fold) if fn else 0.0

            fold_metrics['fold'] = fold_idx
            fold_metrics['repeat'] = r + 1
            all_fold_results.append(fold_metrics)

    mean_metrics = {}
    std_metrics = {}
    for metric_name in metrics:
        values = [fr.get(metric_name, 0.0) for fr in all_fold_results]
        mean_metrics[metric_name] = float(np.mean(values)) if values else 0.0
        std_metrics[metric_name] = float(np.std(values)) if values else 0.0

    return {
        'fold_results': all_fold_results,
        'mean': mean_metrics,
        'std': std_metrics,
        'cv_folds': cv,
        'cv_repeats': repeats,
        'n_total_folds': cv * repeats
    }


def validate_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
) -> Dict[str, float]:
    """
    Compute validation metrics on a single dataset.
    """
    X_array, y_array = _to_numpy_Xy(X, y)

    y_pred = model.predict(X_array)

    metric_functions = {
        'accuracy': lambda: accuracy_score(y_array, y_pred),
        'precision': lambda: precision_score(y_array, y_pred, average='weighted', zero_division=0),
        'recall': lambda: recall_score(y_array, y_pred, average='weighted', zero_division=0),
        'f1': lambda: f1_score(y_array, y_pred, average='weighted', zero_division=0),
        'roc_auc': lambda: roc_auc_score(y_array, model.predict_proba(X_array)[:, 1]) if hasattr(model, 'predict_proba') else 0.0
    }

    results = {}
    for metric_name in metrics:
        try:
            results[metric_name] = float(metric_functions[metric_name]())
        except (AttributeError, IndexError, ValueError):
            results[metric_name] = 0.0

    return results


def generate_confusion_matrix(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Generate and optionally save a confusion matrix.
    """
    X_array, y_array = _to_numpy_Xy(X, y)
    y_pred = model.predict(X_array)
    cm = confusion_matrix(y_array, y_pred)

    if save_path:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return cm


def generate_classification_report(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray]
) -> str:
    """
    Generate a detailed classification report.
    """
    X_array, y_array = _to_numpy_Xy(X, y)
    y_pred = model.predict(X_array)
    return classification_report(y_array, y_pred)


def validate_from_files(
    model_path: str,
    data_path: str,
    target_col: str = 'Fault',
    use_cv: bool = True,
    cv: int = 5,
    scaler_path: Optional[str] = None,
    use_repeated_cv: bool = False,
    cv_repeats: int = 3,
    random_state: int = 42,
    use_smote: bool = True,
    apply_scaling: bool = True,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate a saved model using data from a CSV file.

    NEW:
    - supports repeated CV (cv=10, repeats=3) using use_repeated_cv=True
    - can override SMOTE/scaling flags explicitly
    """
    import joblib

    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    model = joblib.load(model_path)

    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[target_col]), df[target_col]

    # Optional external scaler (legacy)
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        # If you pass an external scaler, we shouldn't refit scaling inside CV
        apply_scaling = False

    if use_cv:
        if use_repeated_cv:
            return cross_validate_model_repeated(
                model=model,
                X=X,
                y=y,
                cv=cv,
                repeats=cv_repeats,
                metrics=metrics,
                random_state=random_state,
                use_smote=use_smote,
                apply_scaling=apply_scaling
            )
        else:
            return cross_validate_model(
                model=model,
                X=X,
                y=y,
                cv=cv,
                metrics=metrics,
                random_state=random_state,
                use_smote=use_smote,
                apply_scaling=apply_scaling
            )
    else:
        return validate_model(model, X, y, metrics=metrics)
