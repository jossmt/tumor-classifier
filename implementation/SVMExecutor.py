from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd

def svm_predict(x_train, y_train, x_test, y_test):
    svmres = svm.SVC(kernel='linear')
    svmres.fit(x_train, y_train)
    y_pred = svmres.predict(x_test)
    y_train_pred = svmres.predict(x_train)

    print('Accuracy on SVM training set: ' + str(accuracy_score(y_train, y_train_pred)))
    print(classification_report(y_train, y_train_pred))

    print('Accuracy on SVM test set: ' + str(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))


def svm_tuned_predict(x_train, y_train, x_test, y_test):
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(x_train, y_train)

    print(grid.best_estimator_)

    y_pred = grid.predict(x_test)
    y_train_pred = grid.predict(x_train)

    print('Accuracy on SVM training set: ' + str(accuracy_score(y_train, y_train_pred)))
    print(classification_report(y_train, y_train_pred))

    print('Accuracy on SVM test set: ' + str(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))