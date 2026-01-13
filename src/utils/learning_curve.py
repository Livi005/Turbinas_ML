from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

def clone_with_random_state(estimator, random_state=None):
    """
    Clona el estimator; si tiene argumento random_state intenta fijarlo.
    """
    est = clone(estimator)
    try:
        est.set_params(random_state=random_state)
    except Exception:
        # fallback si set_params no existe o no acepta random_state
        if hasattr(est, "random_state"):
            setattr(est, "random_state", random_state)

    return est

def learning_curve_with_resampling(
    estimator,
    X,
    y,
    sampler=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    random_state=42,
    verbose=False,
    use_tqdm=False
):
    """
    Genera curva de aprendizaje con resampling opcional en cada fold.
    
    Parámetros
    ----------
    estimator : sklearn estimator
        Modelo que implementa fit/predict.
    X : array-like
        Features.
    y : array-like
        Target binario (clase minoritaria = 1).
    sampler : objeto imblearn o None
        Sampler para oversampling/undersampling. Si es None, no se aplica.
        Ej: SMOTE(), RandomUnderSampler(), None.
    train_sizes : array-like
        Fracciones del conjunto de entrenamiento para evaluar.
    cv : int
        Número de folds estratificados.
    random_state : int
        Semilla para reproducibilidad.
    verbose : bool
        Imprime progreso detallado.
    use_tqdm : bool
        Muestra barra de progreso si tqdm está instalado.
    
    Retorna
    -------
    dict con métricas de aprendizaje y muestra gráfico.
    """
    
    # Configurar scorer F2-Score
    f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)
    
    # Configurar validación cruzada
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Inicializar listas para métricas
    train_f2_scores = [[] for _ in train_sizes]
    val_f2_scores = [[] for _ in train_sizes]
    train_sizes_n = []
    
    # Preparar iteración de folds
    folds = list(skf.split(X, y))
    iter_folds = enumerate(folds, start=1)
    
    if use_tqdm:
        try:
            iter_folds = tqdm(iter_folds, total=len(folds), desc="Folds CV")
        except ImportError:
            warnings.warn("tqdm no instalado, continuando sin barra de progreso")
    
    # Iterar sobre folds
    for fold_idx, (train_idx, val_idx) in iter_folds:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx}/{cv}")
            print(f"{'='*50}")
        
        # Separar datos del fold
        X_train_full = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        y_train_full = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
        
        if verbose:
            print(f"Datos originales -> Train: {len(train_idx)}, Val: {len(val_idx)}")
            print(f"Distribución clase 1 -> Train: {(y_train_full == 1).sum()}, Val: {(y_val == 1).sum()}")
        
        # Iterar sobre tamaños de entrenamiento
        for size_idx, frac in enumerate(train_sizes):
            # Calcular tamaño absoluto
            n_samples = max(2, int(np.floor(frac * len(train_idx))))
            
            # Guardar tamaño en primer fold
            if fold_idx == 1:
                train_sizes_n.append(n_samples)
            
            # Submuestrear datos de entrenamiento
            if n_samples < len(train_idx):
                X_sub, _, y_sub, _ = train_test_split(
                    X_train_full, y_train_full,
                    train_size=n_samples,
                    stratify=y_train_full,
                    random_state=random_state
                )
            else:
                X_sub, y_sub = X_train_full, y_train_full
            
            # Aplicar resampling si se especificó
            if sampler is not None:
                sampler_clone = clone_with_random_state(sampler, random_state)
                try:
                    X_resampled, y_resampled = sampler_clone.fit_resample(X_sub, y_sub)
                except ValueError as e:
                    if verbose:
                        print(f"Advertencia: Resampling falló con {n_samples} muestras: {e}")
                        print("Usando datos originales sin resampling")
                    X_resampled, y_resampled = X_sub, y_sub
            else:
                X_resampled, y_resampled = X_sub, y_sub
            
            if verbose and sampler is not None:
                print(f"  Tamaño {n_samples} -> Resampling: {len(y_sub)} → {len(y_resampled)} muestras")
            
            # Entrenar modelo
            model_clone = clone_with_random_state(estimator, random_state)
            model_clone.fit(X_resampled, y_resampled)
            
            # Calcular F2-Score
            train_f2 = f2_scorer(model_clone, X_resampled, y_resampled)
            val_f2 = f2_scorer(model_clone, X_val, y_val)
            
            # Guardar resultados
            train_f2_scores[size_idx].append(train_f2)
            val_f2_scores[size_idx].append(val_f2)
            
            if verbose:
                print(f"  Tamaño {n_samples:4d} -> Train F2: {train_f2:.4f}, Val F2: {val_f2:.4f}")
    
    # Calcular estadísticas
    train_f2_mean = np.array([np.mean(scores) for scores in train_f2_scores])
    train_f2_std = np.array([np.std(scores) for scores in train_f2_scores])
    val_f2_mean = np.array([np.mean(scores) for scores in val_f2_scores])
    val_f2_std = np.array([np.std(scores) for scores in val_f2_scores])
    
    # Crear gráfico
    plt.figure(figsize=(10, 6))
    x = np.array(train_sizes_n)
    
    # Gráfico principal
    plt.plot(x, train_f2_mean, 'o-', color='blue', linewidth=2, label='Train F2-Score')
    plt.fill_between(x, train_f2_mean - train_f2_std, 
                     train_f2_mean + train_f2_std, alpha=0.2, color='blue')
    
    plt.plot(x, val_f2_mean, 's-', color='red', linewidth=2, label='Validation F2-Score')
    plt.fill_between(x, val_f2_mean - val_f2_std, 
                     val_f2_mean + val_f2_std, alpha=0.2, color='red')
    
    # Configurar gráfico
    plt.xlabel('Tamaño del conjunto de entrenamiento (muestras)', fontsize=12)
    plt.ylabel('F2-Score', fontsize=12)
    
    # Título según tipo de resampling
    if sampler is None:
        title = 'Curva de aprendizaje (sin resampling)'
    else:
        sampler_name = sampler.__class__.__name__
        title = f'Curva de aprendizaje con {sampler_name}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Información resumida
    best_idx = np.argmax(val_f2_mean)
    print("\n" + "="*60)
    print("RESUMEN DE LA CURVA DE APRENDIZAJE")
    print("="*60)
    print(f"Mejor tamaño de entrenamiento: {train_sizes_n[best_idx]} muestras")
    print(f"Mejor F2-Score en validación: {val_f2_mean[best_idx]:.4f} (±{val_f2_std[best_idx]:.4f})")
    print(f"F2-Score correspondiente en train: {train_f2_mean[best_idx]:.4f}")
    
    if sampler is not None:
        print(f"Técnica de resampling: {sampler.__class__.__name__}")
    
    # Retornar resultados
    return {
        "train_sizes": np.array(train_sizes_n),
        "train_f2_mean": train_f2_mean,
        "train_f2_std": train_f2_std,
        "val_f2_mean": val_f2_mean,
        "val_f2_std": val_f2_std,
        "best_size": train_sizes_n[best_idx],
        "best_val_f2": val_f2_mean[best_idx]
    }
