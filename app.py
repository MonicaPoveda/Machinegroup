from flask import Flask, render_template, request

#from SVM import msv_model, scaler
from SVM import (
    msv_model,
    scaler,
    accuracy,
    precision, recall, f1,
    conf_matrix, data_plot_url,
    confusion_plot_url,
    num_records ,
      num_train, num_test, interpretation
)
import LinearRegressionGraddes

from LinearRegressionApplication import (
    data, data_preview, model as model_lr, num_records as lr_num_records,
    coefficient, intercept, plot_url, predict_download_time
)

# Import the SVM graph generation function and trained model
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

    plot_url = LinearRegressionGraddes.generate_plot(
        predicted_hours=hours, predicted_grade=result
    )

    return render_template(
        'linear_regression/conceptsLR.html',
        result=result,
        hours=hours,
        plot_url=plot_url
    )

# APPLICATION OF LINEAR REGRESSION
@app.route("/linear_regression/application", methods=["GET", "POST"])
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
        num_records=lr_num_records,
        data_preview=data_preview,
        data_full=data_full,
        plot_url=plot_url,
        prediction=prediction,
        file_size=file_size,
        error=error,
        coefficient=coefficient,
        intercept=intercept
    )

# LINEAR REGRESSION MENU
@app.route("/linear_regression")
def linear_regression_menu():
    return render_template("linear_regression/linear_regression.html")

# LOGISTIC REGRESSION MENU AND PLACEHOLDER PAGES
@app.route("/logistic_regression")
def logistic_regression_menu():
    return render_template("logistic_regression/logistic_regression.html")

@app.route("/logistic_regression/concepts")
def logistic_regression_concepts():
    return render_template(
        "logistic_regression/placeholder.html",
        page_title="Logistic Regression Concepts",
        page_type="Concepts"
    )

@app.route("/logistic_regression/application")
def logistic_regression_application():
    return render_template(
        "logistic_regression/placeholder.html",
        page_title="Logistic Regression Application",
        page_type="Application"
    )

@app.route("/logistic_regression/evaluation-metrics")
def logistic_regression_metrics():
    return render_template(
        "logistic_regression/placeholder.html",
        page_title="Logistic Regression Evaluation Metrics",
        page_type="Evaluation Metrics"
    )

# USE CASES
@app.route("/use_cases")
def use_cases():
    return render_template("machine_learning/use_cases/use_cases.html")

# CASE 1
@app.route('/case1/')
def case1():
    return render_template('machine_learning/use_cases/case1.html')

# CASE 2
@app.route('/case2/')
def case2():
    return render_template('machine_learning/use_cases/case2.html')

# CASE 3
@app.route('/case3/')
def case3():
    return render_template('machine_learning/use_cases/case3.html')

# CASE 4
@app.route('/case4/')
def case4():
    return render_template('machine_learning/use_cases/case4.html')

# SVM CONCEPTS
@app.route("/SVM/concepts")
def svm_concepts():
    return render_template("svm/conceptsSVM.html")

# SVM MENU
@app.route("/SVM")
def svm_menu():
    return render_template("svm/svm.html")

# SVM APPLICATION
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
            # Get the values from the form
            glucose = float(request.form.get("glucose", "").strip())
            blood_pressure = float(request.form.get("blood_pressure", "").strip())
            bmi = float(request.form.get("bmi", "").strip())
            age = float(request.form.get("age", "").strip())

            # Validate the input values
            if any(value <= 0 for value in [glucose, blood_pressure, bmi, age]):
                error = "All values must be greater than 0."
            else:
                # Create the input data provided by the user
                input_data = [[glucose, blood_pressure, bmi, age]]

                # Standardize the input using the scaler
                # already trained in SVM.py
                input_data_scaled = scaler.transform(input_data)

                # Make the prediction using the trained model
                prediction = msv_model.predict(input_data_scaled)[0]

        except (ValueError, TypeError):
            error = "Please enter valid numerical values."

    return render_template(
        "svm/svmApplication.html",
        # Prediction result
        prediction=prediction,
        # Input values
        glucose=glucose,
        blood_pressure=blood_pressure,
        bmi=bmi,
        age=age,
        # Error
        error=error,
        # Model information
        num_records=num_records,
        num_train=num_train,
        num_test=num_test,
        # Graphics
        data_plot_url=data_plot_url
    )

# SVM EVALUATION METRICS
@app.route("/SVM/EvaluationMetrics")
def svm_evalMetrics():
    return render_template(
        "svm/evalMetrics.html",
        # Metrics
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        # Confusion matrix
        conf_matrix=conf_matrix,
        # Graphics
        confusion_plot_url=confusion_plot_url,
        # Interpretation
        interpretation=interpretation
    )

# LINEAR REGRESSION GRADES
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

    # Generate the graph encoded in Base64
    plot_url = LinearRegressionGraddes.generate_plot(
        predicted_hours=hours,
        predicted_grade=CalculateGradeResult
    )

    return render_template(
        'tempLinearRegression.html',
        result=CalculateGradeResult,
        plot_url=plot_url
    )

# CLASSIFICATION
@app.route('/clasificacion', methods=['GET', 'POST'])
def clasifi():
    resultado = None
    clase_css = ""

    # Generate the initial graph without the user's point
    grafica_base64 = generateSvmPlotBase64()

    if request.method == 'POST':
        # Capture the values from the HTML form
        horas = float(request.form['horas'])
        asistencia = float(request.form['asistencia'])

        # Make the SVM prediction
        prediccion = model_svm.predict([[horas, asistencia]])[0]

        if prediccion == 1:
            resultado = f"Resultado para {horas}h de estudio y {asistencia}% asistencia: ¡ALUMNO APROBADO!"
            clase_css = "aprobado"
        else:
            resultado = f"Resultado para {horas}h de estudio y {asistencia}% asistencia: ALUMNO REPROBADO"
            clase_css = "reprobado"

        # Regenerate the graph including the user's query point
        grafica_base64 = generateSvmPlotBase64(
            horas, asistencia, prediccion
        )

    return render_template(
        'index.html',
        resultado=resultado,
        clase_css=clase_css,
        grafica=grafica_base64
    )

if __name__ == '__main__':
    app.run(debug=True)