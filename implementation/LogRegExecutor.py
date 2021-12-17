from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def log_reg_predict(x_train, y_train, x_test, y_test):
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    y_pred = log_reg.predict(x_test)
    y_train_pred = log_reg.predict(x_train)

    print('Accuracy on LogReg training set: ' + str(accuracy_score(y_train, y_train_pred)))
    print(classification_report(y_train, y_train_pred))

    print('Accuracy on LogReg test set: ' + str(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))
