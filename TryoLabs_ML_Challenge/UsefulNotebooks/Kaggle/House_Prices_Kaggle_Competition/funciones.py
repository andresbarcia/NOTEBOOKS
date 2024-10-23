# Ingesta
import numpy as np
import pandas as pd
import scipy.stats as stats

# Preprocesamiento
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
import matplotlib.gridspec as gridspec
import sweetviz as sv

# Modelación
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor

# Métricas de evaluación
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Otros
import funciones as f
import warnings

warnings.filterwarnings("ignore")


def plot_hist(data, variable, figsize=(12, 5)):
    plt.figure(figsize=figsize)
    sns.histplot(
        data[variable],
        stat="count",
        kde=True,
        color="darkolivegreen",
        lw=0.4,
        alpha=0.5,
    ).set_title(f"Histograma '{variable}'")
    plt.ylabel("Frecuencia")
    plt.axvline(
        data[variable].mean(), color="darkorange", linestyle="--", label="Media"
    )
    plt.axvline(data[variable].median(), color="red", linestyle="--", label="Mediana")
    plt.axvline(data[variable].max(), color="cyan", linestyle="--", label="Máximo")
    plt.axvline(data[variable].min(), color="magenta", linestyle="--", label="Mínimo")
    plt.legend(loc="best")


def distribuciones_continuas(data, figsize=(20, 30), filas=10, columnas=4):
    plt.figure(figsize=figsize)
    for index, column in enumerate(data.columns):
        plt.subplot(filas, columnas, index + 1)
        sns.histplot(x=data[column], kde=True, alpha=0.4, color="darkolivegreen")
        plt.axvline(
            data[column].mean(), color="darkorange", linestyle="--", label="Media"
        )
        plt.axvline(data[column].median(), color="red", linestyle="--", label="Mediana")
        plt.axvline(data[column].max(), color="cyan", linestyle="--", label="Máximo")
        plt.axvline(data[column].min(), color="magenta", linestyle="--", label="Mínimo")
        plt.legend(loc="upper right")
        plt.title(f"Histograma '{column}'")
        plt.ylabel("Frecuencia")
    plt.tight_layout()


def distribuciones_categoricas(data, figsize=(20, 46), filas=15, columnas=3):
    plt.figure(figsize=figsize)
    for index, column in enumerate(data.columns):
        plt.subplot(filas, columnas, index + 1)
        sns.histplot(data[column], color="darkolivegreen", alpha=0.6)
        plt.title(f"Gráfico de barras de '{column}'")
        plt.xticks(rotation=30, ha="right")
        plt.xlabel("Categoría")
        plt.ylabel("Cantidad")
    plt.tight_layout()


def boxplot_target_variable(
    data, cat_or_num_data, figsize=(20, 46), filas=15, columnas=3
):
    plt.figure(figsize=figsize)
    for index, column in enumerate(cat_or_num_data.columns):
        plt.subplot(filas, columnas, index + 1)
        sns.boxplot(x=column, y="SalePrice", data=data)
        plt.title(f"Gráfico de barras de '{column}'")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Categoría")
        plt.ylabel("SalePrice")
    plt.tight_layout()


def graph_missing_values(data, title: str, ax=None):
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)

    if ax is None:
        plt.figure(figsize=(8, 4))
    else:
        plt.sca(ax)

    plt.bar(missing.index, missing.values, color="darkolivegreen", alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Missing Values en {title}")

    for i, count in enumerate(missing):
        plt.text(i, count, str(count), ha="center", va="bottom")
    plt.tight_layout()


def graficar_nulos(data, var: str):
    """
    Ingresa un dataframe y la variable a graficar
    """
    null_counts_genre = data[var].isna().value_counts()
    plt.figure(figsize=(3, 4))
    plt.bar(["No nulos", "Nulos"], null_counts_genre, color=["goldenrod", "red"])
    plt.title(f"Distribución de nulos en {var}")
    plt.ylabel("Cantidad")

    for i, count in enumerate(null_counts_genre):
        plt.text(i, count, str(count), ha="center", va="bottom")


def imputar_cat_moda(data1, data2, var: str):
    data1[var] = np.where(data1[var].isnull(), 0, data1[var])
    data2[var] = np.where(data2[var].isnull(), 0, data2[var])


def modelacion(modelo, X_train, X_test, y_train, y_test):
    print(f"**** Métricas modelo {modelo} ****\n")
    model_tmp_fit = modelo.fit(X_train, y_train)
    y_hat_train = model_tmp_fit.predict(X_train)
    y_hat = model_tmp_fit.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=y_hat_train))
    mae_train = mean_absolute_error(y_true=y_train, y_pred=y_hat_train)
    r2_train = r2_score(y_true=y_train, y_pred=y_hat_train)

    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_hat))
    mae_test = mean_absolute_error(y_true=y_test, y_pred=y_hat)
    r2_test = r2_score(y_true=y_test, y_pred=y_hat)

    print(
        f"{modelo}",
        "\n",
        f"Métricas en train:\n",
        f"RMSE train: {rmse_train}",
        "\n",
        f"MAE train: {mae_train}",
        "\n",
        f"R2 train: {r2_train}",
        "\n",
        "-" * 35,
        "\n" f"Métricas en test:\n",
        f"RMSE test: {rmse_test}",
        "\n",
        f"MAE test: {mae_test}",
        "\n",
        f"R2 test: {r2_test}",
        "\n",
        "_" * 70,
        "\n",
    )

    return model_tmp_fit, y_hat_train, y_hat
