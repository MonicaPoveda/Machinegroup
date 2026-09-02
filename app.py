from flask import Flask, render_template, request
import LinearRegressionGraddes
app = Flask(__name__)


# HOME
@app.route("/")
def home():
    return render_template("home.html")


# MACHINE LEARNING
@app.route("/what-is-ml/")
def what_is_ml():
    return render_template(
        "machine_learning/what_is_ml.html"
    )

#types of machine learning
@app.route("/types/")
def types():
    return render_template(
        "machine_learning/types.html"
    )


@app.route('/LinearRegression/', methods=['GET', 'POST'])
def LRegressionGrades():
    CalculateGradeResult =None
    if request.method == 'POST':
        hours = float(request.form['hours'])
        CalculateGradeResult = LinearRegressionGraddes.calculateGrade(hours)
    return render_template('tempLinearRegression.html', result=CalculateGradeResult)

   # return str(CalculateGradeResult)  # Return the result as a string

if __name__ == '__main__':
   app.run(debug=True)
