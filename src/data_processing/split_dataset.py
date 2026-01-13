import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(
        input_path="src/dataset/gas_turbine_fault_detection.csv",
        output_dir="src/dataset/splits",
        test_size=0.20,
        val_size=0.15,
        random_state=42
    ):
    """
    Divide el dataset en train, validation y test usando división estratificada.
    Guarda los resultados como CSV en output_dir.

    Parámetros:
        input_path (str): Ruta al archivo CSV original.
        output_dir (str): Carpeta donde se guardarán los splits.
        test_size (float): Porcentaje para el conjunto de test.
        X-val_size (float): Porcentaje para validation en relación al train original.
        random_state (int): Semilla para reproducibilidad.
    """

    # Crear carpeta si no existe
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cargar dataset
    df = pd.read_csv(input_path)

    # 2. Verificar valores faltantes
    if df.isnull().sum().any():
        print("⚠️ Advertencia: Se encontraron valores faltantes. Rellenando con forward fill...")
        df = df.fillna(method="ffill")

    # 3. Separar características y etiqueta
    X = df.drop(columns=["Fault"])
    y = df["Fault"]

    # 4. Train/Test estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # # 5. Separar validación a partir de train estratificado
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train, y_train,
    #     test_size=val_size,
    #     stratify=y_train,
    #     random_state=random_state
    # )

    # 6. Reconstruir dataframes completos
    train_df = pd.concat([X_train, y_train], axis=1)
    # val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # 7. Guardar resultados
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    # val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("✅ División completada:")
    print(f"   Train: {train_df.shape}")
    # print(f"   Validation: {val_df.shape}")
    print(f"   Test: {test_df.shape}")
    print(f"✅ Archivos guardados en: {output_dir}")

if __name__ == "__main__":
    split_dataset()
