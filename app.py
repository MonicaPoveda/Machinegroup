from flask import Flask, render_template, request
#from SVM import msv_model, scaler
from SVM import (
    msv_model,
    scaler,
    accuracy,
    precision,
    recall,
    f1,
    conf_matrix,
    data_plot_url,
    confusion_plot_url,
    num_records,
    num_train,
    num_test,
    interpretation
)
import LinearRegressionGraddes


from LinearRegressionApplication import (
    data,
    data_preview,
    model as model_lr,
    num_records,
    coefficient,
    intercept,
    plot_url,
    predict_download_time
)


# Importamos tanto la función de la gráfica como el modelo SVM entrenado
from ModeloClasif import generateSvmPlotBase64, model as model_svm 


app = Flask(__name__)

# HOME
@app.route("/")
def home():
    return render_template("home.html")

# MACHINE LEARNING
@app.route("/what-is-ml/")
def what_is_ml():
    return render_template("machine_learning/what_is_ml.html")

# TYPES OF MACHINE LEARNING
@app.route("/types/")
def types():
    return render_template("machine_learning/types.html")


# CONCEPTION AND INTERACTIVE MODEL
@app.route('/linear_regression/conceptsRL/', methods=['GET', 'POST'])
def linear_regression_concepts():
    result = None
    hours = None
    
    if request.method == 'POST':
        try:
            hours = float(request.form.get('hours'))
            result = LinearRegressionGraddes.calculateGrade(hours)
        except (ValueError, TypeError):
            result = None

    plot_url = LinearRegressionGraddes.generate_plot(predicted_hours=hours, predicted_grade=result)
    
    return render_template(
        'linear_regression/conceptsLR.html',
        result=result,
        hours=hours,
        plot_url=plot_url
    )

#Application of Linear Regression
@app.route("/linear_regression/application",methods=["GET", "POST"])

def linear_regression_application():

    prediction = None
    error = None
    file_size = None

    if request.method == "POST":
        value = request.form.get("file_size", "").strip()
        if not value:
            error = "Please enter a file size."
        else:
            try:
                file_size = float(value)
                if file_size <= 0:
                    error = "File size must be greater than 0."
                else:
                    prediction = predict_download_time(file_size)
                    prediction = round(prediction, 2)
            except ValueError:
                error = "Please enter a valid numeric value."

    # Convert the complete dataset to dictionaries
    data_full = data[["file_size_mb", "download_time_sec"]].to_dict("records")

    return render_template(
        "linear_regression/applicationLR.html",
        num_records=num_records,
        data_preview=data_preview,
        data_full=data_full,
        plot_url=plot_url,
        prediction=prediction,
        file_size=file_size,
        error=error,
        coefficient=coefficient,
        intercept=intercept
    )



@app.route("/use_cases")
def use_cases():
    return render_template("machine_learning/use_cases/use_cases.html")

@app.route('/case1/')
def case1():
    return render_template('machine_learning/use_cases/case1.html')


@app.route('/case2/')
def case2():
    return render_template('machine_learning/use_cases/case2.html')

@app.route('/case3/')
def case3():
    return render_template('machine_learning/use_cases/case3.html')

@app.route('/case4/')
def case4():
        return render_template('machine_learning/use_cases/case4.html')



@app.route("/SVM/application", methods=["GET", "POST"])
def svm_application():

    prediction = None
    error = None

    glucose = None
    blood_pressure = None
    bmi = None
    age = None

    if request.method == "POST":

        try:

            # Obtener los valores del formulario
            glucose = float(
                request.form.get("glucose", "").strip()
            )

            blood_pressure = float(
                request.form.get("blood_pressure", "").strip()
            )

            bmi = float(
                request.form.get("bmi", "").strip()
            )

            age = float(
                request.form.get("age", "").strip()
            )

            # Validar valores
            if any(value <= 0 for value in [
                glucose,
                blood_pressure,
                bmi,
                age
            ]):

                error = "Todos los valores deben ser mayores que 0."

            else:

                # Datos introducidos por el usuario
                input_data = [[
                    glucose,
                    blood_pressure,
                    bmi,
                    age
                ]]

                # Estandarizar utilizando el scaler
                # que ya fue entrenado en SVM.py
                input_data_scaled = scaler.transform(
                    input_data
                )

                # Realizar la predicción
                # utilizando el modelo ya entrenado
                prediction = msv_model.predict(
                    input_data_scaled
                )[0]

        except (ValueError, TypeError):

            error = (
                "Por favor, ingresa valores numéricos válidos."
            )

    return render_template("svm/svmApplication.html",

        # Resultado de predicción
        prediction=prediction,

        # Valores introducidos
        glucose=glucose,
        blood_pressure=blood_pressure,
        bmi=bmi,
        age=age,

        # Error
        error=error,

        # Información del modelo
        num_records=num_records,
        num_train=num_train,
        num_test=num_test,

        # Métricas
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,

        # Matriz
        conf_matrix=conf_matrix,

        # Gráficas
        data_plot_url=data_plot_url,
        confusion_plot_url=confusion_plot_url,

        # Interpretación
        interpretation=interpretation
    )



@app.route('/LinearRegression/', methods=['GET', 'POST'])
def LRegressionGrades():
    CalculateGradeResult = None
    hours = None

    if request.method == 'POST':
        try:
            hours = float(request.form.get('hours'))
            CalculateGradeResult = LinearRegressionGraddes.calculateGrade(hours)
        except (ValueError, TypeError):
            CalculateGradeResult = None

    # Genera la gráfica codificada en Base64
    plot_url = LinearRegressionGraddes.generate_plot(predicted_hours=hours, predicted_grade=CalculateGradeResult)    

    return render_template('tempLinearRegression.html', result=CalculateGradeResult, plot_url=plot_url)




@app.route('/clasificacion', methods=['GET', 'POST'])
def clasifi():
    resultado = None
    clase_css = ""
    # Genera la gráfica inicial sin el punto del usuario
    grafica_base64 = generateSvmPlotBase64()
    
    if request.method == 'POST':
        # Capturar valores del formulario HTML
        horas = float(request.form['horas'])
        asistencia = float(request.form['asistencia'])
        
        # Predicción del SVM
        prediccion = model_svm.predict([[horas, asistencia]])[0]
        
        if prediccion == 1:
            resultado = f"Resultado para {horas}h de estudio y {asistencia}% asistencia: ¡ALUMNO APROBADO!"
            clase_css = "aprobado"
        else:
            resultado = f"Resultado para {horas}h de estudio y {asistencia}% asistencia: ALUMNO REPROBADO"
            clase_css = "reprobado"
            
        # Regenerar la gráfica incluyendo la marca de la consulta del usuario
        grafica_base64 = generateSvmPlotBase64(horas, asistencia, prediccion)
        
    return render_template('index.html', resultado=resultado, clase_css=clase_css, grafica=grafica_base64)


if __name__ == '__main__':
    app.run(debug=True)
    