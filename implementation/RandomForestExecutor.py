from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def random_forest_predict(x_train, y_train, x_test, y_test):
    rf = RandomForestClassifier(n_estimators=2, random_state=42)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    y_train_pred = rf.predict(x_train)

    print('Accuracy on RandomForest training set: ' + str(accuracy_score(y_train, y_train_pred)))
    print(classification_report(y_train, y_train_pred))

    print('Accuracy on RandomForest test set: ' + str(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))
