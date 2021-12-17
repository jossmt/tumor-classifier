from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import random
from matplotlib import pyplot as plt


def knn_predict(x_train, y_train, x_test, y_test, n_neighbours=2):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbours, weights='distance')

    knn_model.fit(x_train, y_train)
    y_pred = knn_model.predict(x_test)
    y_train_pred = knn_model.predict(x_train)

    print('Accuracy on KNN training set: ' + str(accuracy_score(y_train, y_train_pred)))
    print(classification_report(y_train, y_train_pred))

    print('Accuracy on KNN test set: ' + str(accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))


def knn_cluster(X, pca):
    y_pred = KMeans(n_clusters=5).fit_predict(X)

    classes = {}
    for i in range(X.shape[0]):

        print(X[i])
        yi_class = y_pred[i]
        print(yi_class)
        if yi_class in classes:
            classes.update({yi_class: classes.get(yi_class).extend(X[i])})
        else:
            classes[yi_class] = [X[i]]

    for key in classes.keys():
        samples = random.sample(classes[key], 10)
        print('Class: ' + key)
        for sample in samples:
            plt.imshow(pca.inverse_transform(sample))