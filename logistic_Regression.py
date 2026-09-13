import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

data = pd.read_csv("static/datasets/spam_Deteccion.csv")

X = data[["Palabras_Sospechosas", "Cantidad_Links"]]
y = data["Spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

modelo = LogisticRegression()

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

matriz = confusion_matrix(y_test, y_pred)


def clasificar_mensaje(palabras_sospechosas, cantidad_links):
    nuevo_mensaje = pd.DataFrame({
        "Palabras_Sospechosas": [palabras_sospechosas],
        "Cantidad_Links": [cantidad_links]
    })
    prediccion = modelo.predict(nuevo_mensaje)
    probabilidad = modelo.predict_proba(nuevo_mensaje)
    return prediccion[0], probabilidad[0]

