from flask import Flask, render_template, request
import LinearRegressionGraddes

from LinearRegressionApplication import (
    data,
    data_preview,
    model,
    num_records,
    coefficient,
    intercept,
    plot_url,
    predict_download_time
)


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


@app.route("/LinearRegressionApp", methods=["GET", "POST"])
def LRegressionDownload():
    calculateTime_result = None
    error_message = None
    
    if request.method == "POST":
        try:
            file_size = float(request.form["file_size"])
            if file_size < 0:
                error_message = "El tamaño no puede ser negativo"
            elif file_size > 10000:
                error_message = "El tamaño no puede ser mayor a 10000 MB"
            else:
                calculateTime_result = LineaR.calculateTime(file_size)
        except ValueError:
            error_message = "Error: Ingrese un valor numérico válido"
        except Exception as e:
            error_message = f"Error: {str(e)}"
    
    return render_template("linear_regression/applicationLR.html", 
                         result=calculateTime_result,
                         error=error_message)

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


if __name__ == '__main__':
    app.run(debug=True)
    