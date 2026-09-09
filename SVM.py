import io
import base64

import pandas as pd 
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split # Importamos la función para dividir los datos en entrenamiento y prueba
from sklearn.preprocessing import StandardScaler # Importamos la clase para estandarizar los datos

from sklearn.svm import SVC  # Importamos el Clasificador SVM

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score 


import numpy as np

#1. cargamos los datos desde un archivo CSV
df = pd.read_csv("static/datasets/diabetes.csv")

#2. definimos las variables predictoras (X) y la variable objetivo (y)

x = df [[ "Glucose", "BloodPressure", "BMI",  "Age"]].values
y = df["Outcome"].values

#3. Dividimos los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#4. Estandarizamos los datos para que tengan media 0 y desviación estándar 1
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)


#5. creamos el modelo SVM con un Kernel Lineal
msv_model = SVC(kernel='linear', C=1.0)

#6. Entrenamos el modelo SVM con un Kernel Lineal
msv_model.fit(x_train_scaler, y_train)


#7. Hacemos predicciones en el conjunto de prueba
y_pred = msv_model.predict(x_test_scaler)


#8. creamos la matriz de confusión para evaluar el rendimiento del modelo
conf_matrix = confusion_matrix(y_test, y_pred)

#9. visualizamos la matriz de confusión
#plt.figure(figsize=(8, 6))
#sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
#plt.title('Confusion Matrix')
#plt.show()


#10.METRICAS:  evaluamos el modelo utilizando métricas de clasificación

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,zero_division=0)
recall = recall_score( y_test,y_pred,zero_division=0)
f1 = f1_score(y_test,y_pred,zero_division=0)
matrix = confusion_matrix(y_test,y_pred)

print(classification_report(y_test, y_pred))
print (f"exactitud del modelo: {accuracy * 100:.2f}%")
print(f"precision del modelo: {precision * 100:.2f}%")
print(f"recall del modelo: {recall * 100:.2f}%")            
print(f"F1-score del modelo: {f1 * 100:.2f}%")            
print(f"Matriz de confusión:\n{matrix}")

# ============================================================
# 10. FUNCIÓN PARA CONVERTIR UNA GRÁFICA A BASE64
# ============================================================

def figure_to_base64(fig):

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=100
    )

    buffer.seek(0)

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    plt.close(fig)

    return image_base64


# ============================================================
# 11. GRÁFICA DEL CONJUNTO DE DATOS
#
# Glucosa vs IMC
# Diferenciando las clases 0 y 1
# ============================================================

fig_data, ax = plt.subplots(
    figsize=(8, 5)
)

sns.scatterplot(
    data=df,
    x="Glucose",
    y="BMI",
    hue="Outcome",
    style="Outcome",
    s=70,
    ax=ax
)

ax.set_title(
    "Distribución de pacientes según Glucosa e IMC"
)

ax.set_xlabel(
    "Glucosa (mg/dL)"
)

ax.set_ylabel(
    "IMC (kg/m²)"
)

ax.legend(
    title="Resultado"
)

fig_data.tight_layout()

data_plot_url = figure_to_base64(
    fig_data
)


# ============================================================
# 12. GRÁFICA DE LA MATRIZ DE CONFUSIÓN
# ============================================================

fig_matrix, ax = plt.subplots(
    figsize=(6, 5)
)

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=ax
)

ax.set_title(
    "Matriz de Confusión"
)

ax.set_xlabel(
    "Clase predicha"
)

ax.set_ylabel(
    "Clase real"
)

ax.set_xticklabels(
    ["Sano (0)", "Diabético (1)"]
)

ax.set_yticklabels(
    ["Sano (0)", "Diabético (1)"]
)

fig_matrix.tight_layout()

confusion_plot_url = figure_to_base64(
    fig_matrix
)


# ============================================================
# 13. INFORMACIÓN DE LOS DATOS
# ============================================================

num_records = len(df)

num_train = len(x_train)

num_test = len(x_test)


# ============================================================
# 14. INTERPRETACIÓN DE LOS RESULTADOS
# ============================================================

interpretation = (
    f"El modelo SVM obtuvo una exactitud del "
    f"{accuracy * 100:.2f}%, lo que significa que clasificó "
    f"correctamente aproximadamente "
    f"{accuracy * 100:.2f}% de los pacientes del conjunto de prueba. "
    f"La precisión fue de {precision * 100:.2f}%, mientras que el "
    f"recall alcanzó {recall * 100:.2f}%. "
    f"El F1-score fue de {f1 * 100:.2f}%, mostrando un equilibrio "
    f"entre precisión y recall."
)