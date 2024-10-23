# House Prices - Advanced Regression Techniques :house:

<img src="https://www.rocketmortgage.com/resources-cmsassets/RocketMortgage.com/Article_Images/Large_Images/Stock-Neighborhood-Development-AdobeStock307010071%20copy.jpg">

## Tabla de contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto-clipboard)
2. [Evaluación](#evaluación-chart_with_upwards_trend)
3. [Herramientas Utilizadas](#herramientas-utilizadas-wrench)
4. [Estructura del Proyecto](#estructura-del-proyecto-open_file_folder)
5. [Cómo usar este proyecto](#cómo-usar-este-proyecto-question)
6. [Contenido del Jupyter notebook](#contenido-del-jupyter-notebook-page_facing_up)
7. [Modelos Utilizados](#modelos-utilizados-computer)
8. [Resultados](#resultados-bar_chart)


### Descripción del Proyecto :clipboard:
Este proyecto utiliza el conjunto de datos disponible en Kaggle (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) para realizar un análisis de datos utilizando Python. El objetivo principal es predecir el precio de venta para las casas aplicando técnicas de aprendizaje automático.

### Evaluación :chart_with_upwards_trend:
La métrica que se busca mejorar es el Root-Mean-Squared-Error (RMSE). Este valor es una métrica comúnmente utilizada en machine learning para evaluar la calidad de un modelo de regresión. Se utiliza para medir la diferencia entre los valores reales y las predicciones del modelo, y se expresa en la misma unidad que la variable objetivo.

### Herramientas Utilizadas :wrench:
- Python 3.9.17
- Bibliotecas de análisis de datos: Pandas, NumPy.
- Bibliotecas de visualización: Matplotlib, Seaborn.
- Biblioteca de aprendizaje automático: scikit-learn.

### Estructura del Proyecto :open_file_folder:
- **train.csv:** Archivo CSV que contiene los datos de entrenamiento.
- **test.csv:** Archivo CSV que contiene los datos de validación.
- **house_prices.ipynb:** Un Jupyter notebook que contiene el código Python para el análisis de datos.
- **funciones.py:** Archivo Python que contiene las funciones utilizadas para este proyecto.
- **submission.csv:** Archivo CSV que contiene las predicciones para el archivo `test.csv` de acuerdo a las instrucciones proporcionadas por Kaggle.

### Cómo usar este proyecto :question:
1. Asegúrate de tener instalado Python 3.9.17 en tu sistema.
2. Descarga el conjunto de datos desde Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
3. Coloca los archivos CSV descargados (`train.csv`, `test.csv`) en la misma carpeta que este proyecto.
4. Abre el Jupyter notebook `house_prices.ipynb` y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

### Contenido del Jupyter notebook :page_facing_up:
El Jupyter notebook proporciona un análisis completo de los datos, que incluye:
- Exploración de datos: Resumen estadístico, visualización de datos, identificación de valores nulos, etc.
- Preprocesamiento de datos: Limpieza de datos, manejo de valores faltantes, codificación de variables categóricas, etc.
- Análisis de características: Visualización de relaciones entre los atributos.
- Modelado y predicción: Entrenamiento de modelos de aprendizaje automático para predecir el precio de las casas.
- Evaluación del modelo: Evaluación del Root-Mean-Squared-Error (RMSE) y rendimiento del modelo.

### Modelos Utilizados :computer:
- K-Nearest Neighbors Regressor
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- LightGBM Regressor
- Voting Regressor

### Resultados :bar_chart:
Se evaluaron todos los modelos utilizando la métrica Root-Mean-Squared-Error (RMSE), y los resultados son los siguientes:

- K-Nearest Neighbors Regressor: RMSE: 41011
- Random Forest Regressor: RMSE: 32324
- AdaBoost Regressor: RMSE: 33261
- Gradient Boosting Regressor: RMSE: 30395
- LightGBM Regressor: RMSE: 31170
- Voting Regressor: RMSE: 29661


Para la selección del mejor modelo se consideró obtener el menor valor de RMSE y adicionalmente tener un bajo valor de MAE ya que esta métrica indica claramente en cuánto puede variar el valor de una casa. Por lo que el modelo seleccionado es un VotingRegressor que integra los modelos RandomForestRegressor, GradientBoostingRegressor y LGBMRegressor, todos en su tercera iteración.
  Con esta combinación se obtiene un RMSE de 29661.228 y un MAE de 15832.778 dólares.

