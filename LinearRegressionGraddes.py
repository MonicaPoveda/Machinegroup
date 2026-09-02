import pandas as pd   #funciona para dataframes
import matplotlib.pyplot as plt  #Importa Matplotlib, que permite crear gráficos y visualizaciones
import io #Importa el módulo de entrada/salida de Python para manipular buffers de texto o bytes en memoria
import base64 #Importa el módulo para codificar datos binarios (como imágenes) a texto en formato Base64

from sklearn.linear_model import LinearRegression #Importa únicamente el algoritmo de Regresión Lineal
#desde la librería de aprendizaje automático scikit-learn.

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}
df = pd.DataFrame(data) #Convierte el diccionario en un DataFrame de Pandas (una tabla con filas y columnas)

# Create a linear regression model
x=df[["Study Hours"]] #variable independiente - Extrae la columna de horas de estudio como la variable independiente
y=df[["Final Grade"]] #variable dependiente -Extrae la columna de notas como la variable dependiente (lo que queremos predecir)

#varibale para almacenar el modelo

model = LinearRegression() #Crea una instancia (objeto) del modelo de regresión lineal vacío
model.fit(x, y)  #funcion para entrenar el modelo (pasa las variables independiente y dependiente)
#Ejecuta el algoritmo de entrenamiento. El modelo calcula matemáticamente la línea recta 
#que mejor se ajusta a la relación entre x e y (encuentra la pendiente e intercepción).


def calculateGrade(hours):  #funcion recibe como parametro las horas de estudio
    # Predict the final grade based on study hours
    result = model.predict([[hours]]) [0]  #predice la nota final en base a las horas de estudio
    #0=Toma el primer valor del resultado (ya que predict retorna una lista/arreglo con la respuesta).
    return result #Devuelve el valor numérico de la nota estimada.


