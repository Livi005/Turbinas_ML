"""Model training and persistence utilities."""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import joblib

def save_model(
    model: Any,
    model_name: str,
    save_path: str = 'src/models/',
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Save a trained model and its metadata.
    
    Parameters:
    -----------
    model : Any
        Trained model to save
    model_name : str
        Name identifier for the model
    save_path : str, default='src/models/'
        Directory to save the model
    metadata : dict, optional
        Additional metadata to save (hyperparameters, metrics, etc.)
        
    Returns:
    --------
    dict
        Dictionary with paths to saved model and metadata files
        
    Examples:
    ---------
    >>> paths = save_model(model, 'logistic_regression', metadata={'accuracy': 0.95})
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(save_path, model_filename)
    joblib.dump(model, model_path)
    
    # Prepare metadata
    model_metadata = {
        'model_name': model_name,
        'model_path': model_path,
        'timestamp': timestamp,
        'training_date': datetime.now().isoformat(),
    }
    
    # Add model-specific attributes if available
    if hasattr(model, 'get_params'):
        model_metadata['hyperparameters'] = model.get_params()
    
    # Add custom metadata
    if metadata:
        model_metadata.update(metadata)
    
    # Save metadata
    metadata_filename = f"{model_name}_{timestamp}_metadata.json"
    metadata_path = os.path.join(save_path, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)
    
    return {
        'model_path': model_path,
        'metadata_path': metadata_path
    }


def load_model(
    model_name: str,
    model_path: Optional[str] = None
) -> Any:
    """
    Load a saved model.
    
    Parameters:
    -----------
    model_name : str
        Name identifier for the model
    model_path : str, optional
        Full path to the model file. If None, searches in src/models/
        
    Returns:
    --------
    Any
        Loaded model
        
    Raises:
    -------
    FileNotFoundError
        If model file is not found
        
    Examples:
    ---------
    >>> model = load_model('logistic_regression', 'src/models/logistic_regression_20240101_120000.joblib')
    """
    if model_path is None:
        # Search for the most recent model with this name
        models_dir = 'src/models/'
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Find all models with this name
        matching_files = [
            f for f in os.listdir(models_dir)
            if f.startswith(model_name) and f.endswith('.joblib')
        ]
        
        if not matching_files:
            raise FileNotFoundError(
                f"No model found with name '{model_name}' in {models_dir}"
            )
        
        # Get the most recent one (by filename timestamp)
        matching_files.sort(reverse=True)
        model_path = os.path.join(models_dir, matching_files[0])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    return model

