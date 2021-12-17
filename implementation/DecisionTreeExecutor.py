from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report


def decision_tree_predict(x_train, y_train, x_test, y_test):
    tree_params = {
        'criterion': 'entropy'
    }
    clf = tree.DecisionTreeClassifier(**tree_params)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_train_pred = clf.predict(x_train)

    print('Accuracy on DecisionTree training set: ' + str(accuracy_score(y_train, y_train_pred)))
    print(classification_report(y_train, y_train_pred))

    print('Accuracy on DecisionTree test set: ' + str(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))
