"""Utility functions for loading and preparing data."""

import os
import pandas as pd
from typing import Tuple


def load_split(
    split_name: str,
    data_dir: str = "src/dataset/splits"
) -> pd.DataFrame:
    """
    Load a dataset split (train, validation, or test).
    
    Parameters:
    -----------
    split_name : str
        Name of the split to load ('train', 'validation', or 'test')
    data_dir : str
        Directory containing the split files
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset split
        
    Raises:
    -------
    FileNotFoundError
        If the split file doesn't exist
    ValueError
        If split_name is not one of the valid options
    """
    valid_splits = ['train', 'validation', 'test']
    if split_name not in valid_splits:
        raise ValueError(f"split_name must be one of {valid_splits}, got '{split_name}'")
    
    file_path = os.path.join(data_dir, f"{split_name}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = "Fault"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing features and target
    target_col : str
        Name of the target column
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Tuple of (features, target)
        
    Raises:
    -------
    KeyError
        If target_col is not found in the dataframe
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

