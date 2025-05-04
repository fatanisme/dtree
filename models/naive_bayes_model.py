from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def train_and_predict_nb(sepal_length, sepal_width, petal_length, petal_width):
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    pred_label = target_names[pred]

    # Accuracy
    accuracy = model.score(X_test, y_test)

    return pred_label, accuracy
