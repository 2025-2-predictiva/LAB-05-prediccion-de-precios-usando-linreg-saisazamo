#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import pandas as pd
import numpy as np
import json
import os
import gzip
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

# =========================================================
# STEP 1 — PREPROCESSING
# =========================================================
def load_and_preprocess(path_train, path_test, current_year=2021):
    """Load, clean and preprocess data according to instructions."""

    train = pd.read_csv(path_train, compression="zip")
    test = pd.read_csv(path_test, compression="zip")

    # Create Age
    train["Age"] = current_year - train["Year"]
    test["Age"] = current_year - test["Year"]

    # Drop unwanted
    train.drop(columns=["Year", "Car_Name"], inplace=True)
    test.drop(columns=["Year", "Car_Name"], inplace=True)

    return train, test


# =========================================================
# STEP 2 — SPLIT X/Y
# =========================================================
def split_xy(train, test, target="Present_Price"):
    """Split into X and y."""

    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_test, y_test


# =========================================================
# STEP 3 — PIPELINE
# =========================================================
def build_pipeline(categorical_cols, numeric_cols):
    """Build the preprocessing + model pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", MinMaxScaler(), numeric_cols)
        ],
        remainder="drop"
    )

    model = LinearRegression()

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("select", SelectKBest(score_func=f_regression)),
        ("model", model)
    ])

    return pipe


# =========================================================
# STEP 4 — HYPERPARAMETER OPTIMIZATION
# =========================================================
def optimize_pipeline(pipe, X_train, y_train):
    """Run GridSearchCV to optimize hyperparameters."""

    param_grid = {
        "select__k": range(1, 12),
        'model__fit_intercept': [True, False],
        'model__positive': [True, False]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    return grid


# =========================================================
# STEP 5 — SAVE MODEL
# =========================================================
def save_model(grid, output_path="files/models/model.pkl.gz"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        pickle.dump(grid, f)
    print(f"Model saved at: {output_path}")


# =========================================================
# STEP 6 — METRIC COMPUTATION
# =========================================================
def compute_metrics(y_true, y_pred, dataset_name):
    """Create metrics dictionary."""

    return {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mad": median_absolute_error(y_true, y_pred),
    }


def save_metrics(metrics, output_path="files/output/metrics.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")


# =========================================================
# MASTER FUNCTION (RUN EVERYTHING)
# =========================================================
def main():
    # Load
    train, test = load_and_preprocess(
        "files/input/train_data.csv.zip",
        "files/input/test_data.csv.zip"
    )

    # Split
    X_train, y_train, X_test, y_test = split_xy(train, test)

    categorical_cols = ["Fuel_Type", 'Selling_type', "Transmission"]
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    # Pipeline
    pipe = build_pipeline(categorical_cols, numeric_cols)

    # Optimize
    grid = optimize_pipeline(pipe, X_train, y_train)
    print("Best params:", grid.best_params_)

    # Save model
    save_model(grid)

    # Predictions
    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    # Metrics
    metrics = []
    metrics.append(compute_metrics(y_train, y_train_pred, "train"))
    metrics.append(compute_metrics(y_test, y_test_pred, "test"))

    save_metrics(metrics)
    print("Metrics saved successfully.")


# =========================================================
# RUN SCRIPT
# =========================================================
if __name__ == "__main__":
    main()

