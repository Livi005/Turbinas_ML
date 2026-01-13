"""SMOTE-based data augmentation for imbalanced datasets."""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
from imblearn.over_sampling import SMOTE


def apply_smote(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    sampling_strategy: Union[str, float, dict] = 'auto',
    random_state: int = 42,
    **smote_params
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to balance classes.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    sampling_strategy : str, float, or dict, default='auto'
        Strategy to use for resampling. Options:
        - 'auto': resample to balance classes
        - float: ratio of minority to majority class
        - dict: specify exact number of samples per class
    random_state : int, default=42
        Random seed for reproducibility
    **smote_params
        Additional parameters to pass to SMOTE (e.g., k_neighbors)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Augmented feature matrix and target vector
        
    Examples:
    ---------
    >>> X_aug, y_aug = apply_smote(X_train, y_train)
    >>> X_aug, y_aug = apply_smote(X_train, y_train, sampling_strategy=0.8, k_neighbors=5)
    """
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        feature_names = X.columns.tolist()
    else:
        X_array = X
        feature_names = None
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    # Initialize SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        **smote_params
    )
    
    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X_array, y_array)
    
    return X_resampled, y_resampled


def augment_dataset(
    df: pd.DataFrame,
    target_col: str = 'Fault',
    sampling_strategy: Union[str, float, dict] = 'auto',
    random_state: int = 42,
    return_dataframe: bool = True,
    **smote_params
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray]]:
    """
    High-level function to augment a full dataset using SMOTE.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features and target
    target_col : str, default='Fault'
        Name of the target column
    sampling_strategy : str, float, or dict, default='auto'
        Strategy to use for resampling
    random_state : int, default=42
        Random seed for reproducibility
    return_dataframe : bool, default=True
        If True, return pandas DataFrame/Series. If False, return numpy arrays.
    **smote_params
        Additional parameters to pass to SMOTE
        
    Returns:
    --------
    Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray]]
        Augmented features and target. Type depends on return_dataframe parameter.
        
    Examples:
    ---------
    >>> df_aug = augment_dataset(train_df)
    >>> X_aug, y_aug = augment_dataset(train_df, return_dataframe=False)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Apply SMOTE
    X_resampled, y_resampled = apply_smote(
        X, y,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        **smote_params
    )
    
    if return_dataframe:
        # Convert back to DataFrame/Series
        X_df = pd.DataFrame(X_resampled, columns=X.columns)
        y_series = pd.Series(y_resampled, name=target_col)
        return X_df, y_series
    else:
        return X_resampled, y_resampled

