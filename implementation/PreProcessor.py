import pandas as pd
from imageio import imread
import numpy as np
from skimage.exposure import exposure
from skimage.feature import hog
from skimage.io import imshow
from skimage.transform import resize
from sklearn.decomposition import PCA
import os
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from collections import Counter
from matplotlib import pyplot as plt
from skimage.filters import prewitt_h, prewitt_v
from sklearn.preprocessing import MinMaxScaler
import shutil

le = preprocessing.LabelEncoder()
pca = PCA(100)
scaler = MinMaxScaler()


def pre_process_data_hog_pca():
    if os.path.isfile('../dataset/X_HOG_PCA.pickle'):
        print('Started reading from files')
        X = pd.read_pickle('../dataset/X_HOG_PCA.pickle')
        print('Finished reading from files')
        return X

    df = pd.read_csv('../dataset/label.csv')

    X = pd.DataFrame()

    for index, row in df.iterrows():
        img_gray = imread('../dataset/image/' + row['file_name'], as_gray=True)

        fd, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)

        data_rescaled = scaler.fit_transform(hog_image)

        img_transformed = pca.fit_transform(data_rescaled)

        # Start plot section
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        # _, axs = plt.subplots(1, 2, figsize=(12, 12))
        # axs = axs.flatten()
        # imgs = [img_gray, hog_image_rescaled]
        # for img, ax in zip(imgs, axs):
        #     ax.imshow(img)
        # plt.show()
        # End plot section

        features = np.reshape(img_transformed, (512 * 100))
        if np.any(np.isnan(features)):
            print('features creating nans')

        X = X.append(pd.Series(features).T, ignore_index=True)
        print("\rCompleted {:.2f}".format((index / df.shape[0]) * 100), end="")

    X.to_pickle('../dataset/X_HOG_PCA.pickle')
    return X


def pre_process_data_old(do_pca=False, do_edge=False):
    if os.path.isfile('../dataset/X.pickle'):
        print('Started reading from files')
        X = pd.read_pickle('../dataset/X.pickle')
        print('Finished reading from files')
        return X

    df = pd.read_csv('../dataset/label.csv')

    X = pd.DataFrame()

    # features = np.reshape(img_gray, (512*512))
    # print(features.shape)

    # print(img_gray.shape)
    # img_transformed = pca.transform(img_gray)
    # print(img_transformed.shape)
    # img_transformed_features = np.reshape(img_transformed, (512*32))
    # print(img_transformed_features.shape)
    # print(np.sum(pca.explained_variance_ratio_) )

    # Retrieving the results of the image after Dimension reduction.
    # temp = pca.inverse_transform(img_transformed)
    # print(temp.shape)
    # temp = np.reshape(temp, (512,512))
    # print(temp.shape)
    # plt.imshow(temp)

    for index, row in df.iterrows():
        img_gray = imread('../dataset/image/' + row['file_name'], as_gray=True)
        if (do_pca):
            img_transformed = pca.fit_transform(img_gray)
        elif (do_edge):
            # calculating vertical edges using prewitt kernel
            img_transformed = prewitt_v(img_gray)
        else:
            img_transformed = img_gray

        features = np.reshape(img_transformed, (2560))
        X = X.append(pd.Series(features).T, ignore_index=True)
        print("\rCompleted {:.2f}".format((index / df.shape[0]) * 100), end="")

    if (do_pca):
        X.to_pickle('../dataset/X_pca.pickle')
    elif (do_edge):
        X.to_pickle('../dataset/X_edge.pickle')
    else:
        X.to_pickle('../dataset/X.pickle')
    return X


def resolve_imbalances_smote(X, Y):
    print(Counter(Y))
    oversample = SMOTE()
    X, Y = oversample.fit_resample(X, Y)
    print(Counter(Y))
    return X, Y


def y_binary():
    df = pd.read_csv('../dataset/label.csv')
    return (df['label'] != 'no_tumor').astype(int)


def y_multiclass():
    df = pd.read_csv('../dataset/label.csv')
    le.fit(df['label'])
    return le.transform(df['label'])


def invert_multiclass(Y):
    return le.inverse_transform(Y)

def generate_tensorflow_input_binary():
    df = pd.read_csv('../dataset/label.csv')

    for index, row in df.iterrows():
        label = 'tumor' if row['label'] != 'no_tumor' else 'no_tumor'
        # shutil.copyfile('../dataset/image/' + row['file_name'], '../dataset/binary_tf/' + label + '/' + row['file_name'])
        img = imread('../dataset/image/' + row['file_name'])
        print(img.shape)
        break

def generate_tensorflow_input_multiclass():
    df = pd.read_csv('../dataset/label.csv')
    for index, row in df.iterrows():
        shutil.copyfile('../dataset/image/' + row['file_name'], '../dataset/multiclass_tf/' + row['label'] + '/' + row['file_name'])

def generate_tensorflow_input_binary_test():
    df = pd.read_csv('../dataset/test/label.csv')

    for index, row in df.iterrows():
        label = 'tumor' if row['label'] != 'no_tumor' else 'no_tumor'
        shutil.copyfile('../dataset/test/image/' + row['file_name'], '../dataset/test/binary_tf/' + label + '/' + row['file_name'])

def generate_tensorflow_input_multiclass_test():
    df = pd.read_csv('../dataset/test/label.csv')
    for index, row in df.iterrows():
        shutil.copyfile('../dataset/test/image/' + row['file_name'], '../dataset/test/multiclass_tf/' + row['label'] + '/' + row['file_name'])