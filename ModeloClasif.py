import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Modo no interactivo para el servidor
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # Importamos el Clasificador SVM

# 1. Datos de entrenamiento (Horas de estudio, % Asistencia)
data = {
    "Horas_Estudio": [10, 15, 3, 12, 8, 14, 5, 18, 7, 11, 2, 16, 9, 4, 13, 6, 17, 1, 20, 5],
    "Asistencia":    [85, 90, 40, 80, 70, 95, 50, 98, 60, 75, 30, 92, 65, 45, 88, 55, 89, 20, 95, 60],
    "Estado":        [1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0] 
    # 1 = Aprobado (Azul), 0 = Reprobado (Rojo)
}
df = pd.DataFrame(data)

# Variables predictoras (X) y variable objetivo (y)
X = df[["Horas_Estudio", "Asistencia"]].values
y = df["Estado"].values

# 2. Entrenamos el modelo SVM con un Kernel Lineal
# Usamos un margen suave (C=1.0) para permitir clasificaciones óptimas
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

# 3. Función del formulario para clasificar un nuevo alumno
def classifyStudent(horas, asistencia):
    # Predice si el alumno aprueba (1) o reprueba (0)
    prediccion = model.predict([[horas, asistencia]])[0]
    # Retorna un texto amigable y el código numérico
    resultado_texto = "APROBADO" if prediccion == 1 else "REPROBADO"
    return resultado_texto, int(prediccion)

# 4. Función para generar la gráfica con el hiperplano y enviarla en Base64
def generateSvmPlotBase64(user_horas=None, user_asistencia=None, user_pred=None):
    plt.figure(figsize=(9, 6))
    
    # Dibujar los datos existentes
    aprobados = df[df["Estado"] == 1]
    reprobados = df[df["Estado"] == 0]
    
    plt.scatter(aprobados["Horas_Estudio"], aprobados["Asistencia"], color='blue', label='Aprobados (Datos)', s=50)
    plt.scatter(reprobados["Horas_Estudio"], reprobados["Asistencia"], color='red', label='Reprobados (Datos)', s=50)
    
    # Crear la malla para dibujar las líneas divisorias (Hiperplano y Márgenes)
    ax = plt.gca()
    xlim = (0, 22)
    ylim = (10, 105)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.decision_function(xy).reshape(xx.shape)
    
    # Dibujar el Hiperplano (0) y los Márgenes (-1, 1)
    ax.contour(xx, yy, Z, colors='black', levels=[-1, 0, 1], alpha=0.6, linestyles=['--', '-', '--'])
    
    # Resaltar los Vectores de Soporte (los puntos clave que definen la frontera)
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=120,
               facecolors='none', edgecolors='black', linewidths=1.5, label='Vectores de Soporte')
    
    # Si el usuario mandó datos desde el formulario, graficamos "Su Punto"
    if user_horas is not None and user_asistencia is not None:
        color_punto = 'darkblue' if user_pred == 1 else 'darkred'
        plt.scatter(user_horas, user_asistencia, color=color_punto, marker='X', s=250, 
                    edgecolors='black', label=f'Tu Consulta ({user_horas}h, {user_asistencia}%)')
    
    plt.title('Clasificación de Alumnos con Máquina de Vectores de Soporte (SVM)')
    plt.xlabel('Horas de Estudio')
    plt.ylabel('% de Asistencia')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Convertir gráfica a string Base64 para el HTML
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"
