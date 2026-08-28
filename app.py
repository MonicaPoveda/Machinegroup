from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/types/')
def types():
    return render_template('types.html')

@app.route('/What_is_ML/')
def what_is_ml():
    return render_template('what_is_ml.html')
