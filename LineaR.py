import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
#.\.venv\Scripts\python.exe -m pip install scikit-learn

# ===== IMPORTAR DATASET DESDE CSV =====
#print(" Importando dataset...")
df = pd.read_csv('Data/dataset.csv')
#print(f"Dataset importado con {len(df)} registros")

# Mostrar primeros 5 registros
#print("\n Primeros 5 registros:")
#print(df.head())

# ===== VARIABLES =====
x = df[["File_Size_MB"]]        # Variable independiente (X)
y = df[["Download_Time_Sec"]]    # Variable dependiente (Y)

# ===== MODELO =====
model = LinearRegression()
model.fit(x, y)  # Entrenar el modelo

print(f"\n Modelo entrenado:")
print(f"   Pendiente (m): {model.coef_[0][0]:.4f}")
print(f"   Intercepto (b): {model.intercept_[0]:.2f}")
print(f"   R²: {model.score(x, y):.4f}")


# ===== FUNCIÓN PARA PREDECIR =====
def calculateTime(file_size_mb):
    """
    Predice el tiempo de descarga basado en el tamaño del archivo.
    """
    result = model.predict([[file_size_mb]])[0][0]
    return round(result, 2)

# ===== INFORMACIÓN DEL MODELO =====
def get_model_info():
    """
    Devuelve la información del modelo y del dataset.
    """
    r2 = model.score(x, y)
    return {
        'pendiente': round(model.coef_[0][0], 3),
        'intercepto': round(model.intercept_[0], 2),
        'r2': round(r2, 4),
        'total_records': len(df),
        'csv_file': 'Data/dataset.csv'
    }

# ===== PROBAR =====
if __name__ == "__main__":
    test_size = 500
    result = calculateTime(test_size)
    print(f"\n Prueba: Para un archivo de {test_size} MB")
    print(f" Tiempo estimado: {result} segundos")