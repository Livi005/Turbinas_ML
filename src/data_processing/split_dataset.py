import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    splits_dir="src/dataset/splits",
    final_test_size=0.50,
    random_state=42
):
    """
    Divide el conjunto test existente en:
      - test.csv  -> validación (durante entrenamiento)
      - testF.csv -> test final real (evaluación final)

    NO modifica train.csv.

    Parámetros:
        splits_dir (str): Carpeta donde están los splits actuales.
        final_test_size (float): Proporción del test original que irá a testF.
        random_state (int): Semilla para reproducibilidad.
    """

    test_path = os.path.join(splits_dir, "test.csv")
    testF_path = os.path.join(splits_dir, "testF.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No se encontró {test_path}")

    # 1. Cargar test actual (que ahora será validación + test final)
    df_test = pd.read_csv(test_path)

    if "Fault" not in df_test.columns:
        raise ValueError("La columna 'Fault' no está presente en test.csv")

    # 2. Separar features y target
    X = df_test.drop(columns=["Fault"])
    y = df_test["Fault"]

    # 3. Split estratificado: validation (test) / testF
    X_val, X_testF, y_val, y_testF = train_test_split(
        X,
        y,
        test_size=final_test_size,
        stratify=y,
        random_state=random_state
    )

    # 4. Reconstruir dataframes
    val_df = pd.concat([X_val, y_val], axis=1)
    testF_df = pd.concat([X_testF, y_testF], axis=1)

    # 5. Guardar archivos
    val_df.to_csv(test_path, index=False)      # test.csv → validación
    testF_df.to_csv(testF_path, index=False)   # testF.csv → test final

    print("✅ División del conjunto test completada:")
    print(f"   Validation (test.csv): {val_df.shape}")
    print(f"   Test Final (testF.csv): {testF_df.shape}")
    print(f"📁 Archivos guardados en: {splits_dir}")


if __name__ == "__main__":
    split_dataset()
