"""Model training and persistence utilities."""

from typing import Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    model_name: str = 'logistic_regression',
    save_path: str = 'src/models/',
    **hyperparams
) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    model_name : str, default='logistic_regression'
        Name identifier for the model
    save_path : str, default='src/models/'
        Directory to save the model
    **hyperparams
        Additional hyperparameters for LogisticRegression
        (e.g., C, max_iter, penalty, solver)
        
    Returns:
    --------
    LogisticRegression
        Trained model
        
    Examples:
    ---------
    >>> model = train_logistic_regression(X_train, y_train)
    >>> model = train_logistic_regression(X_train, y_train, C=0.1, max_iter=1000)
    """
    # Convert to numpy arrays if needed
    if isinstance(X_train, pd.DataFrame):
        X_array = X_train.values
    else:
        X_array = X_train
    
    if isinstance(y_train, pd.Series):
        y_array = y_train.values
    else:
        y_array = y_train
    
    # Default hyperparameters
    default_params = {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'lbfgs'
    }
    
    # Merge with user-provided hyperparameters
    params = {**default_params, **hyperparams}
    
    # Initialize and train model
    model = LogisticRegression(**params)
    model.fit(X_array, y_array)
    
    return model