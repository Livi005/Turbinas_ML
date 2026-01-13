"""Model validation and evaluation utilities."""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.augmentation.smote_augment import apply_smote

import matplotlib.pyplot as plt
import seaborn as sns


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
    
    According to the research methodology:
    1. For each fold iteration, 4 folds are used for training and 1 for validation
    2. SMOTE is applied ONLY to the training data of that fold (not the whole dataset)
    3. The validation fold remains unchanged
    4. StandardScaler is fitted on training data and applied to both training and validation
    5. Model is trained on augmented training data and evaluated on original validation fold
    
    Parameters:
    -----------
    model : Any
        Sklearn-compatible model to validate
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    cv : int, default=5
        Number of folds for cross-validation
    metrics : list of str, default=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        Metrics to compute
    random_state : int, default=42
        Random seed for reproducibility
    use_smote : bool, default=True
        Whether to apply SMOTE augmentation to training data in each fold
    apply_scaling : bool, default=True
        Whether to apply StandardScaler to features
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'fold_results': List of metrics for each fold
        - 'mean': Mean metrics across folds
        - 'std': Standard deviation of metrics across folds
        - 'cv_folds': Number of folds used
        
    Examples:
    ---------
    >>> cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    """
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
    else:
        X_array = X
        feature_names = None
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Store results for each fold
    fold_results = []
    
    # Metric computation functions
    metric_functions = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'roc_auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array)):
        # Step 1: Split into training and validation folds
        X_train_fold = X_array[train_idx]
        X_val_fold = X_array[val_idx]
        y_train_fold = y_array[train_idx]
        y_val_fold = y_array[val_idx]  # Keep validation fold unchanged
        
        # Step 2: Apply SMOTE ONLY to training data of this fold
        if use_smote:
            X_train_aug, y_train_aug = apply_smote(
                X_train_fold,
                y_train_fold,
                random_state=random_state
            )
        else:
            X_train_aug = X_train_fold
            y_train_aug = y_train_fold
        
        # Step 3: Apply StandardScaler fitted on training data
        if apply_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_aug)
            X_val_scaled = scaler.transform(X_val_fold)  # Apply same transformation to validation
        else:
            X_train_scaled = X_train_aug
            X_val_scaled = X_val_fold
        
        # Step 4: Train model on augmented and scaled training data
        model_copy = type(model)(**model.get_params())
        model_copy.fit(X_train_scaled, y_train_aug)
        
        # Step 5: Predict on validation fold (original, not augmented)
        y_pred_fold = model_copy.predict(X_val_scaled)
        
        # Step 6: Compute metrics
        fold_metrics = {}
        for metric_name in metrics:
            if metric_name == 'roc_auc':
                # ROC-AUC needs probability predictions
                try:
                    y_pred_proba = model_copy.predict_proba(X_val_scaled)[:, 1]
                    fold_metrics[metric_name] = metric_functions[metric_name](y_val_fold, y_pred_proba)
                except (AttributeError, IndexError):
                    fold_metrics[metric_name] = 0.0
            else:
                fold_metrics[metric_name] = metric_functions[metric_name](y_val_fold, y_pred_fold)
        
        fold_metrics['fold'] = fold_idx + 1
        fold_results.append(fold_metrics)
    
    # Step 7: Compute mean and std across folds
    mean_metrics = {}
    std_metrics = {}
    
    for metric_name in metrics:
        metric_values = [fold[metric_name] for fold in fold_results]
        mean_metrics[metric_name] = np.mean(metric_values)
        std_metrics[metric_name] = np.std(metric_values)
    
    return {
        'fold_results': fold_results,
        'mean': mean_metrics,
        'std': std_metrics,
        'cv_folds': cv
    }


def validate_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
) -> Dict[str, float]:
    """
    Compute validation metrics on a single dataset.
    
    Parameters:
    -----------
    model : Any
        Trained model to validate
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    metrics : list of str, default=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        Metrics to compute
        
    Returns:
    --------
    dict
        Dictionary of computed metrics
        
    Examples:
    ---------
    >>> metrics = validate_model(model, X_val, y_val)
    """
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    # Make predictions
    y_pred = model.predict(X_array)
    
    # Metric computation functions
    metric_functions = {
        'accuracy': lambda: accuracy_score(y_array, y_pred),
        'precision': lambda: precision_score(y_array, y_pred, average='weighted', zero_division=0),
        'recall': lambda: recall_score(y_array, y_pred, average='weighted', zero_division=0),
        'f1': lambda: f1_score(y_array, y_pred, average='weighted', zero_division=0),
        'roc_auc': lambda: roc_auc_score(y_array, model.predict_proba(X_array)[:, 1]) if hasattr(model, 'predict_proba') else 0.0
    }
    
    # Compute metrics
    results = {}
    for metric_name in metrics:
        try:
            results[metric_name] = metric_functions[metric_name]()
        except (AttributeError, IndexError, ValueError) as e:
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
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    save_path : str, optional
        Path to save the confusion matrix plot
        
    Returns:
    --------
    np.ndarray
        Confusion matrix array
        
    Examples:
    ---------
    >>> cm = generate_confusion_matrix(model, X_val, y_val, 'confusion_matrix.png')
    """
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    # Make predictions
    y_pred = model.predict(X_array)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_array, y_pred)
    
    # Plot and save if path provided
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
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
        
    Returns:
    --------
    str
        Classification report as string
        
    Examples:
    ---------
    >>> report = generate_classification_report(model, X_val, y_val)
    >>> print(report)
    """
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    # Make predictions
    y_pred = model.predict(X_array)
    
    # Generate report
    report = classification_report(y_array, y_pred)
    
    return report


def validate_from_files(
    model_path: str,
    data_path: str,
    target_col: str = 'Fault',
    use_cv: bool = True,
    cv: int = 5,
    scaler_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate a saved model using data from a CSV file.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model file
    data_path : str
        Path to CSV file with data
    target_col : str, default='Fault'
        Name of target column
    use_cv : bool, default=True
        Whether to use cross-validation
    cv : int, default=5
        Number of folds for cross-validation (if use_cv=True)
    scaler_path : str, optional
        Path to saved StandardScaler. If provided, will scale the data before validation.
        
    Returns:
    --------
    dict
        Validation results
        
    Examples:
    ---------
    >>> results = validate_from_files('model.joblib', 'data.csv', use_cv=True, scaler_path='scaler.joblib')
    """
    # Load model
    import joblib
    model = joblib.load(model_path)
    
    # Load data
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[target_col]), df[target_col]
    
    # Apply scaler if provided
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    
    # Validate
    if use_cv:
        results = cross_validate_model(model, X, y, cv=cv, use_smote=True, apply_scaling=False)
    else:
        results = validate_model(model, X, y)
    
    return results

