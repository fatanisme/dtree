from flask import Flask, render_template, request
from models.decision_tree_model import train_and_predict
from models.naive_bayes_model import train_and_predict_nb

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/decision-tree', methods=['GET', 'POST'])
def decision_tree():
    prediction = None
    accuracy = None

    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            prediction, accuracy = train_and_predict(sepal_length, sepal_width, petal_length, petal_width)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('decision_tree.html', prediction=prediction, accuracy=accuracy)


@app.route('/naive-bayes', methods=['GET', 'POST'])
def naive_bayes():
    prediction = None
    accuracy = None

    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            prediction, accuracy = train_and_predict_nb(sepal_length, sepal_width, petal_length, petal_width)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('naive_bayes.html', prediction=prediction, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
