from flask import Flask, render_template, request
from LinearRegressionGraddes import calculateGrade, generate_plot
import LinearRegressionGraddes

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

    plot_url = generate_plot(predicted_hours=hours, predicted_grade=result)
    
    return render_template(
        'linear_regression/conceptsLR.html',
        result=result,
        hours=hours,
        plot_url=plot_url
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

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/case1/')
def case1():
    return render_template('machine_learning/use_cases/case1.html')


@app.route('/case2/')
def case2():
    return render_template('machine_learning/use_cases/case2.html')

