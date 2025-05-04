import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os

# Fungsi untuk melatih model dan prediksi berdasarkan input
def train_and_predict(sepal_length, sepal_width, petal_length, petal_width):
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Decision Tree
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Prediksi berdasarkan input user
    pred = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    pred_label = target_names[pred]

    # Hitung akurasi
    accuracy = clf.score(X_test, y_test)

    # Visualisasi pohon
    save_tree_plot(clf, iris.feature_names, target_names)

    return pred_label, accuracy


def save_tree_plot(clf, feature_names, class_names):
    # Membuat folder static jika belum ada
    if not os.path.exists('static'):
        os.makedirs('static')

    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.savefig("static/tree_plot.png")
    plt.close()
