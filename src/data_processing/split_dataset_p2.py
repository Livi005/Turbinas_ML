import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_problem2_dataset(
    input_csv="src/dataset/gas_turbine_fault_detection_simulated3.csv",
    output_dir="src/dataset/splits_p2",
    train_size=0.70,
    final_test_size=0.50,
    random_state=42
):
    """
    Crea splits para el Problema 2 (clasificación del tipo de falla).

    - Lee el dataset simulado.
    - Filtra SOLO las filas con Fault == 1 (porque el tipo de falla solo aplica ahí).
    - Estratifica por la columna de tipo de falla (FaultMode / Fault Mode / etc.).
    - Genera:
        train.csv  -> entrenamiento
        test.csv   -> validación durante entrenamiento
        testF.csv  -> test final (evaluación final real)
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"No se encontró el dataset: {input_csv}")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # --- Detectar nombres de columnas (robusto) ---
    # Fault
    if "Fault" not in df.columns:
        raise ValueError(
            "No se encontró la columna 'Fault' en el CSV. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # Fault Mode puede venir con espacio o sin espacio, etc.
    candidate_faultmode_cols = ["FaultMode", "Fault Mode", "fault_mode", "faultmode"]
    faultmode_col = None
    for c in candidate_faultmode_cols:
        if c in df.columns:
            faultmode_col = c
            break

    if faultmode_col is None:
        raise ValueError(
            "No se encontró ninguna columna de tipo de falla. "
            f"Busqué: {candidate_faultmode_cols}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # --- Filtrar SOLO fallas para Problema 2 ---
    df = df[df["Fault"] == 1].copy()

    if df.empty:
        raise ValueError("Tras filtrar Fault==1, el dataset quedó vacío.")

    # Target multiclase
    y = df[faultmode_col]

    # Verificar clases
    n_classes = y.nunique(dropna=True)
    if n_classes < 2:
        raise ValueError(
            f"La columna '{faultmode_col}' tiene {n_classes} clase(s) tras Fault==1. "
            "No se puede entrenar un clasificador multiclase con eso."
        )

    # (1) Split: train vs resto (validación+testF)
    df_train, df_rest = train_test_split(
        df,
        train_size=train_size,
        stratify=y,
        random_state=random_state
    )

    # (2) Split del resto: test (validación) vs testF (final)
    y_rest = df_rest[faultmode_col]
    df_test, df_testF = train_test_split(
        df_rest,
        test_size=final_test_size,
        stratify=y_rest,
        random_state=random_state
    )

    # Guardar
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")     # VALIDACIÓN
    testF_path = os.path.join(output_dir, "testF.csv")   # TEST FINAL

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_testF.to_csv(testF_path, index=False)

    # Resumen
    print("✅ Splits Problema 2 creados (solo Fault==1):")
    print(f"Input: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Columna tipo de falla usada: '{faultmode_col}'")
    print(f"Train:  {df_train.shape} | clases: {df_train[faultmode_col].nunique()}")
    print(f"Test:   {df_test.shape}  | clases: {df_test[faultmode_col].nunique()}  (validación)")
    print(f"TestF:  {df_testF.shape} | clases: {df_testF[faultmode_col].nunique()} (final)")
    print("\nArchivos:")
    print(f" - {train_path}")
    print(f" - {test_path}")
    print(f" - {testF_path}")


if __name__ == "__main__":
    split_problem2_dataset()
