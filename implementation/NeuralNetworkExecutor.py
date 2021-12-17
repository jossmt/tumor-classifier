from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier


def neural_network_predict(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_pred = clf.predict(x_test)

    print(y_train[:10])
    print(y_train_pred[:10])

    print('Accuracy on NeuralNetwork training set: ' + str(accuracy_score(y_train, y_train_pred)))
    print(classification_report(y_train, y_train_pred))

    print(y_test.shape)
    print(y_pred.shape)

    print('Accuracy on NeuralNetwork test set: ' + str(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))


# Try with Keras:
# https://www.kaggle.com/vedicyadav/project-1-my-brain-tumor-detection
# https://www.kaggle.com/nikitanikonov/brain-tumor-detection