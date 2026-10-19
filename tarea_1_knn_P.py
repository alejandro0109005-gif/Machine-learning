# %% [markdown]
# Importamos las librerias necesarias

# %%
# Importamos as bibliotecas necessárias
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score, recall_score, confusion_matrix

from imblearn.over_sampling import SMOTE

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# %% [markdown]
# Importamos los datos dados
# 

# %%
# Importamos los datos 
df = pd.read_stata("Base_deficit_y_muertos_112017_conesposos.dta")
df.head(10)

# %% [markdown]
# Renombramos varias de las columnas para facilitar su lectura y comprensión 

# %%
# Renombramos las columnas para que sean más fáciles de usar 
df = df.rename(columns= {
    "h13_12_d_2" : "Dress", "h5_12_d_2" : "In_out_chair", "h15d_12_d" : "Walk", 
    "h17d_12_d" : "Eat", "h16d_12_d" : "Groom", "h19d_12_d" : "Toilet", "h7_12_d_2": "Stairs",
    "h11_12_d_2" : "Lift", "h27c_12_d_2" : "Shop", "h26c_12_d_2" : "Meals", "h28c_12_d_2" : "Meds", 
    "h29c_12_d_2" : "Finance", "c64_12_d" : "Lost_weight", "c1_12_d" : "Health", "c2a_12_d" : "H_change",
    "c73_12_d" : "Bed", "c49_8_12_d" : "Tired", "h3_12_d_2" : "Walk_out", "c49_2_12_d" : "Effort", 
    "c49_1_12_d" : "Depressed", "c49_4_12_d" : "No_happy", "c49_5_12_d" : "Lone", "c49_9_12_d" : "No_Energy",
    "c4_12_d" : "High_BP", "c22a_12_d" : "Heart_attack", "c25b_12_d" : "CHF", "c26_12_d" : "Stroke",
    "c12_12_menos10_d" : "Cancer", "c6_12_d" : "Diabetes", "c32_12_d" : "Arthritis", "c19_12_d" : "CLD", 
    "e1b_12_d" : "Memory", "c69a_12_d" : "Grip", "fallecido_15" : "Dead", "c70_12_d" : "Anorexia", 
    "c50b_12_d" : "Excercise"})
df.head(10)

# %% [markdown]
# Creamos un dataframe nuevo con las columnas que nos interesan para hacer nuestro modelo, incluyendo el output y los inputs

# %%
# Creamos un dataframe nuevo con las columnas que usaremos para output e input
df_new = df[["Dress", "In_out_chair", "Walk", "Eat", "Groom", "Toilet", "Stairs", "Lift", "Shop",
        "Meals", "Meds", "Finance", "Lost_weight", "Health", "H_change", "Bed", "Tired", "Walk_out",
        "Effort", "Depressed", "No_happy", "Lone", "No_Energy", "High_BP", "Heart_attack", "CHF", 
        "Stroke", "Cancer", "Diabetes", "Arthritis", "CLD", "Memory", "Grip", "Anorexia", "Excercise", "Dead"]]
df_new.head(10)

# %% [markdown]
# Cada una de nuestras variables indican: dificultad (valor 1) o no (valor cero), para:
# 
# Vestirse, levantarse de una silla, caminar, comer, arreglarse, ir al baño,
# levantar cosas pesadas, hacer las compras, cocinar sus alimentos, tomar sus
# medicamentos, manejar sus finanzas, así como pérdida de peso inexplicable,
# problemas de salud, cambios de salud, dificultad para levantarse de la cama,
# sentirse cansado, dificultad para caminar, sentir que todo es un esfuerzo,
# estar o no deprimido, ser infeliz, sentirse solo, sentirse sin energía, tener
# desde el último seguimiento presión arterial alta, ataque cardíaco, falla
# crónica del corazón, embolia, cáncer, diabetes, artritis, problemas
# pulmonares, fallas de memoria, dificultad para tomar objetos, si hay reporte
# de muerte entre el 2012 y 2015, tener anorexia y tener dificultad para
# ejercitarse. 

# %% [markdown]
# De este nuevo dataframe limpiamos los valores faltantes 

# %%
# Eliminamos las filas con valores nulos 
df_new = df_new.dropna()
df_new.head(10)

# %% [markdown]
# Seleccionamos las variables que ocuparemos como inputs y el output

# %%
# Seleccionamos las variables que vamos a usar de inputs y el output que es la variable "Dead"
X_input = df_new[["Dress", "In_out_chair", "Walk", "Eat", "Groom", "Toilet", "Stairs", "Lift", "Shop",
        "Meals", "Meds", "Finance", "Lost_weight", "Health", "H_change", "Bed", "Tired", "Walk_out",
        "Effort", "Depressed", "No_happy", "Lone", "No_Energy", "High_BP", "Heart_attack", "CHF", 
        "Stroke", "Cancer", "Diabetes", "Arthritis", "CLD", "Memory", "Grip", "Anorexia", "Excercise"]]
y = df_new["Dead"]


# %% [markdown]
# Veamos nuestros inputs

# %%
# Visualizamos la variable X_input para asegurarnos de no necesitar hacer un escalamiento de los datos
X_input.head(10)

# %% [markdown]
# Veamos el input de cada observación

# %%
# Ahora veamos y 
y.head(10)

# %% [markdown]
# Dividimos el conjunto de datos en entrenamiento y prueba para el modelo 

# %%
# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_input, y, test_size=0.20, random_state=42)



# %% [markdown]
# Ajustamos el modelo knn con los diferentes modelos de k y obtenemos la precisión para cada k 

# %%
# Ajustamos el modelo KNN con los datos de entrenamiento y variaremos k 
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for k in k_values:
    knn_normal = KNeighborsClassifier(n_neighbors = k)
    knn_normal.fit(X_train, y_train)
    y_pred = knn_normal.predict(X_test)
    precision = precision_score(y_test, y_pred)
    print(f"Precisión del modelo con k={k}: {precision:.4f}")


# %% [markdown]
# Hacemos lo mismo pero ahora para la sensitividad

# %%
# Ahora hacemos lo mismo que anteriormente pero para la sensitividad
for k in k_values:
    knn_normal = KNeighborsClassifier(n_neighbors = k)
    knn_normal.fit(X_train, y_train)
    y_pred = knn_normal.predict(X_test)
    recall = recall_score(y_test, y_pred)
    print(f"Sensitividad del modelo con k={k}: {recall:.4f}")

# %% [markdown]
# Al obtener estos resultados podemos observar que para algunos valores de k la precisión es excelnte, sin embargo la sensitividad más alta apenas llega a 20%. Esto ocurre por el desbalanceo en el conjunto de datos de salida, pues tenemos pocas observaciones con el marcador 1 a comparación del marcador 0

# %% [markdown]
# Solo para comprobar un poco lo que ya veniamos viendo, obtengamos las matrices de confusión 

# %%
# Como los datos estan desbalanceados, vamos a ver la matriz de confusión para ver cuántos casos positivos y negativos se predicen correctamente
for k in k_values:
    knn_normal = KNeighborsClassifier(n_neighbors = k)
    knn_normal.fit(X_train, y_train)
    y_pred = knn_normal.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Matriz de confusión del modelo con k={k}:\n{cm}\n")

# %% [markdown]
# Curva de ROC

# %%
# Obtengamos la grafica de ROC para ver el desempeño del modelo
# Calculamos las probabilidades de predicción
probabilidades = knn_normal.predict_proba(X_test)[:, 1]
# Calculamos los puntos de la curva ROC
fpr, tpr, _ = roc_curve(y_test, probabilidades)
roc_auc = auc(fpr, tpr)
# Graficamos la curva ROC
sns.set(style = "whitegrid")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color = "darkorange", lw = 2, label = f"Curva ROC (área = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC del modelo KNN sin modificación de los datos")
plt.legend(loc = "lower right")
plt.show()


# %% [markdown]
# Observemos que la curva de ROC está muy lejos de ser perfecta para poder decir que el modelo clasifica de manera correcta, confirmando lo que ya númericamente habíamos calculado

# %% [markdown]
# Habiendo visto esto podemos observar que en general nuestro modelo no es ni mucho menos preciso, pues se clasifican mal para cada k 

# %% [markdown]
# Por esta razón, primero probemos una estrategia de modificación del umbral de votación para ir obteniendo proporciones de vecinos que votan por la clase minoritaria para ir variando los pesos, variando el umbral para cada valor de k.

# %%
# Calculamos la sensitividad del modelo variando el umbral de decisión para ver si mejora la sensitividad del modelo
for k in k_values:
    knn_umbral = KNeighborsClassifier(n_neighbors = k)
    knn_umbral.fit(X_train, y_train)
    probabilidades1 = knn_umbral.predict_proba(X_test)[:, 1]
    # Variamos el umbral para ver si mejora la sensitividad del modelo
    for umbral_personalizado in [0.20, 0.25, 0.30, 0.35, 0.40]:
        predicciones_sensibles = (probabilidades1 >= umbral_personalizado).astype(int)
        recall = recall_score(y_test, predicciones_sensibles)
        print(f"Sensitividad del modelo con k={k} y umbral={umbral_personalizado}: {recall:.4f}")
        

# %% [markdown]
# Con esto podemos ver que la sensitividad para varias combinaciones de k y umbral sube, sin embargo sigue sin ser muy grande, siendo el valor mas grande: 41.67%

# %% [markdown]
# Veamos ahora que pasa con la precisión 

# %%
# Probemos ahora para la precision 
for k in k_values:
    knn_umbral = KNeighborsClassifier(n_neighbors = k)
    knn_umbral.fit(X_train, y_train)
    probabilidades2 = knn_umbral.predict_proba(X_test)[:, 1]
    # Variamos el umbral para ver si mejora la precisión del modelo
    for umbral_personalizado in [0.20, 0.25, 0.30, 0.35, 0.40]:
        predicciones_sensibles = (probabilidades2 >= umbral_personalizado).astype(int)
        precision_2 = precision_score(y_test, predicciones_sensibles)
        print(f"Precisión del modelo con k={k} y umbral={umbral_personalizado}: {precision_2:.4f}")

# %% [markdown]
# Notemos que nuestra precisión bajo, siendo el mayor valor: 80%, sin embargo los valores mayores de la precisión con un k y umbral determinado no coinciden con los mayores valores en sensitividad, es por esto que no podemos encontrar una combinación "optima", dejandonos ver que una métrica sube, pero la otra baja

# %% [markdown]
# Por esta razón probemos con otro metódo de SMOTE, para ver si al generar datos sinteticos podemos balancear los datos y lograr obtener mejores métricas 

# %%
# Habiendo visto esto, apliquemos un oversampling para balancear los datos y veamos si mejora la sensitividad del modelo.
smote = SMOTE(random_state = 42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# %% [markdown]
# Entrenamos al modelo knn con estos nuevos datos balanceados y obtenemos la sensitividad

# %%
# Ahora volvamos a entrenar el modelo KNN con los datos balanceados y veamos si mejora la sensitividad del modelo.
for k in k_values:
    knn_smote = KNeighborsClassifier(n_neighbors = k)
    knn_smote.fit(X_train_resampled, y_train_resampled)
    y_pred = knn_smote.predict(X_test)
    recall = recall_score(y_test, y_pred)
    print(f"Sensitividad del modelo con k={k} después de oversampling: {recall:.4f}")
    


# %% [markdown]
# Notemos que la sensitividad sí aumento respecto a la medición inicial y a la técnica que usamos anteriormente, sin embargo no logramos un resultado tan bueno como nos gustaría

# %% [markdown]
# Veamos que pasa con la precisión 

# %%
# Veamos ahora la precisión del modelo después de aplicar oversampling
for k in k_values:
    knn_smote = KNeighborsClassifier(n_neighbors = k)
    knn_smote.fit(X_train_resampled, y_train_resampled)
    y_pred = knn_smote.predict(X_test)
    precision = precision_score(y_test, y_pred)
    print(f"Precisión del modelo con k={k} después de oversampling: {precision:.4f}")

# %% [markdown]
# Curva de ROC

# %%
# Grafiquemos la curva ROC del modelo después de aplicar oversampling
# Calculamos las probabilidades de predicción
probabilidades_oversampling = knn_smote.predict_proba(X_test)[:, 1]
fpr_oversampling, tpr_oversampling, _ = roc_curve(y_test, probabilidades_oversampling)
roc_auc_oversampling = auc(fpr_oversampling, tpr_oversampling)
# Graficamos la curva ROC
sns.set(style = "whitegrid")
plt.figure(figsize=(8, 6))
plt.plot(fpr_oversampling, tpr_oversampling, color = "darkorange", lw = 2, label = f"Curva ROC (área = {roc_auc_oversampling:.2f})")
plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC del modelo KNN después de oversampling")
plt.legend(loc = "lower right")
plt.show()


# %% [markdown]
# Como vimos, aunque balanceamos con datos sinteticos nuestro conjunto de datos, la precisión y la sensibilidad son demasiado bajas, confirmado por la curva de ROC (aunque subio el área respecto de las otras pruebas que hicimos), por ende probemos cambiando la métrica para medir distancia en knn que puede ser mejor para este caso de variables binarias para ambos casos (con oversampling y sin oversampling).
# 

# %% [markdown]
# Precisión sin oversampling

# %%
# Probemos la metrica jaccard sin oversampling y calculemos la precisión del modelo
for k in k_values:
    knn_jaccard = KNeighborsClassifier(n_neighbors = k, metric = "jaccard")
    knn_jaccard.fit(X_train, y_train)
    y_pred = knn_jaccard.predict(X_test)
    precision = precision_score(y_test, y_pred)
    print(f"Métrica Jaccard del modelo con k={k} sin oversampling: {precision:.4f}")

# %% [markdown]
# Notemos que con está nueva métrica logramos excelentes resultados de precisión, sin embargo esto tampoco es tan bueno, pues podríamos estar sufriendo de overfitting, da modo que revisemos que pasa con la sensitividad.

# %%
# Probemos la metrica jaccard sin oversampling y calculemos la sensitividad del modelo
for k in k_values:
    knn_jaccard = KNeighborsClassifier(n_neighbors = k, metric = "jaccard")
    knn_jaccard.fit(X_train, y_train)
    y_pred = knn_jaccard.predict(X_test)
    recall = recall_score(y_test, y_pred)
    print(f"Métrica Jaccard del modelo con k={k} sin oversampling: {recall:.4f}")

# %% [markdown]
# Curva de ROC 

# %%
# Grafiquemos la curva de ROC del modelo con la metrica jaccard sin oversampling
# Calculamos las probabilidades de predicción
probabilidades_sin_oversampling = knn_jaccard.predict_proba(X_test)[:, 1]
fpr_sin_oversampling, tpr_sin_oversampling, _ = roc_curve(y_test, probabilidades_sin_oversampling)
roc_auc_sin_oversampling = auc(fpr_sin_oversampling, tpr_sin_oversampling)

# Graficamos la curva ROC
sns.set(style = "whitegrid")
plt.figure(figsize=(8, 6))
plt.plot(fpr_sin_oversampling, tpr_sin_oversampling, color = "darkorange", lw = 2, label = f"Curva ROC (área = {roc_auc_sin_oversampling:.2f})")
plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC del modelo KNN con métrica Jaccard sin oversampling")
plt.legend(loc = "lower right")
plt.show()

# %% [markdown]
# Nuevamente volvemos a obtener lo mismo que ya veniamos visualizando: Una métrica sube mucho, mientras que la otra baja 
# 
# Así, probemos igual jaccard pero sobre los datos con oversampling y obtengamos la precisión

# %%
# Veamos ahora con oversampling y la metrica jaccard si mejora la precisión del modelo
for k in k_values:
    knn_jaccard_oversampled = KNeighborsClassifier(n_neighbors = k, metric = "jaccard")
    knn_jaccard_oversampled.fit(X_train_resampled, y_train_resampled)
    y_pred = knn_jaccard_oversampled.predict(X_test)
    precision = precision_score(y_test, y_pred)
    print(f"Métrica Jaccard del modelo con k={k} después de oversampling: {precision:.4f}")

# %% [markdown]
# Desde aquí ya podemos ver que la precisión bajo muchisimo, siendo no un muy buen indicador de que nuestro modelo mejoró

# %%
# Veamos ahora con oversampling y la metrica jaccard si mejora la sensitividad del modelo
for k in k_values:
    knn_jaccard_oversampled = KNeighborsClassifier(n_neighbors = k, metric = "jaccard")
    knn_jaccard_oversampled.fit(X_train_resampled, y_train_resampled)
    y_pred = knn_jaccard_oversampled.predict(X_test)
    recall = recall_score(y_test, y_pred)
    print(f"Métrica Jaccard del modelo con k={k} después de oversampling: {recall:.4f}")

# %% [markdown]
# Curva de ROC
# 

# %%
# Grafiquemos la curva de ROC del modelo con la metrica jaccard sin oversampling
# Calculamos las probabilidades de predicción
probabilidades_con_oversampling = knn_jaccard_oversampled.predict_proba(X_test)[:, 1]
fpr_con_oversampling, tpr_con_oversampling, _ = roc_curve(y_test, probabilidades_con_oversampling)
roc_auc_con_oversampling = auc(fpr_con_oversampling, tpr_con_oversampling)

# Graficamos la curva ROC
sns.set(style = "whitegrid")
plt.figure(figsize=(8, 6))
plt.plot(fpr_con_oversampling, tpr_con_oversampling, color = "darkorange", lw = 2, label = f"Curva ROC (área = {roc_auc_con_oversampling:.2f})")
plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC del modelo KNN con métrica Jaccard con oversampling")
plt.legend(loc = "lower right")
plt.show()

# %% [markdown]
# Bien, el mismo efecto que anteriormente hemos visto, aunque está vez no subió tanto la sensitividad. Es por eso que hagamos una última prueba, combinando todo lo que hemos usado anteriormente para ver si tenemos mejores resultados 

# %% [markdown]
# Veamos la sensitividad

# %%
# Finalmente juntemos todo lo anterior y veamos si mejora la sensitividad del modelo con oversampling y la metrica jaccard
for k in k_values:
    knn_combinado = KNeighborsClassifier(n_neighbors = k, metric = "jaccard")
    knn_combinado.fit(X_train_resampled, y_train_resampled)
    probabilidades = knn_combinado.predict_proba(X_test)[:, 1]
    for umbral_personalizado in [0.20, 0.25, 0.30, 0.35, 0.40]:
        predicciones_sensibles = (probabilidades >= umbral_personalizado).astype(int)
        recall = recall_score(y_test, predicciones_sensibles)
        print(f"Sensitividad del modelo con k={k} y umbral={umbral_personalizado}: {recall:.4f}")

# %% [markdown]
# Ahora la precisión

# %%
# Finalmente juntemos todo lo anterior y veamos si mejora la precisión del modelo con oversampling y la metrica jaccard
for k in k_values:
    knn_combinado = KNeighborsClassifier(n_neighbors = k, metric = "jaccard")
    knn_combinado.fit(X_train_resampled, y_train_resampled)
    probabilidades = knn_combinado.predict_proba(X_test)[:, 1]
    for umbral_personalizado in [0.20, 0.25, 0.30, 0.35, 0.40]:
        predicciones_sensibles = (probabilidades >= umbral_personalizado).astype(int)
        precision_2 = precision_score(y_test, predicciones_sensibles)
        print(f"Precisión del modelo con k={k} y umbral={umbral_personalizado}: {precision_2:.4f}")

# %% [markdown]
# Veamos por último la matriz de confusión 

# %%
# Obtengamos la matriz de confusión 
for k in k_values:
    knn_combinado = KNeighborsClassifier(n_neighbors = k, metric = "jaccard")
    knn_combinado.fit(X_train_resampled, y_train_resampled)
    y_pred = knn_combinado.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Matriz de confusión del modelo con k={k} después de oversampling:\n{cm}\n")

# %% [markdown]
# Curva de ROC 

# %%
# Grafiquemos la curva de ROC del modelo con la metrica jaccard sin oversampling
# Calculamos las probabilidades de predicción
probabilidades_combinadas = knn_combinado.predict_proba(X_test)[:, 1]
fpr_combinado, tpr_combinado, _ = roc_curve(y_test, probabilidades_combinadas)
roc_auc_combinado = auc(fpr_combinado, tpr_combinado)

# Graficamos la curva ROC
sns.set(style = "whitegrid")
plt.figure(figsize=(8, 6))
plt.plot(fpr_combinado, tpr_combinado, color = "darkorange", lw = 2, label = f"Curva ROC (área = {roc_auc_combinado:.2f})")
plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC del modelo KNN combinando oversampling y métrica Jaccard")
plt.legend(loc = "lower right")
plt.show()

# %% [markdown]
# Finalmente con esta última técnica podemos observar el mismo fénomeno, pues logramos subir la sensitividad lo más posible, pero la precisión quedo extremadamente baja, pudiendo confirmar de manera general que al tener datos desbalanceados como los que tenemos, al subir una métrica, otra baja, ya sea poco o drasticamente, por lo que quizá haría falta aplicar otro tipo de técnicas para poder obtener mejores resultados en ambas métricas 


