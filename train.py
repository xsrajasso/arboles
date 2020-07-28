# basado en https://gogul09.github.io/software/image-classification-python
# basado en https://github.com/satuelisa/BarkBeetle/blob/master/train.py


#librerias necesarias:
#scikit-learn, opencv, mahotas

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from random import random, shuffle
import numpy as np
import mahotas
import cv2
import os

#RED NEURONAL
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


fixed_size = tuple((76, 76)) # las muestras son 76 por 76 pixeles
count = 75 # how many to take per class
bins = 8

def fd_hu_moments(image): # feature-descriptor-1: Hu Moments
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image): # feature-descriptor-2: Haralick Texture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None): # feature-descriptor-3: Color Histogram
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

labels = []
features = []
targets = ['green', 'yellow', 'red', 'leafless']
counts = defaultdict(int)

for label in targets:
    #listing = list(os.scandir(f'individual/original/{label}'))
    listing = list(os.scandir(f'dataset/train/{label}'))
    shuffle(listing)
    for entry in listing:
        if entry.path.endswith('.png') and entry.is_file():
            if counts[label] < count:
                filename = entry.path
                image = cv2.imread(filename)
                image = cv2.resize(image, fixed_size)
                fv_hu_moments = fd_hu_moments(image)
                fv_haralick = fd_haralick(image)
                fv_histogram  = fd_histogram(image)
                global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
                labels.append(label)
                features.append(global_feature)
                counts[label] += 1

scaler = MinMaxScaler(feature_range = (0, 1)) # normalizador
rescaled_features = scaler.fit_transform(features) # normalizar caracteristicas

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(rescaled_features),
                                                                  np.array(labels),
                                                                  test_size = 0.3) # 30 % para pruebas 

#clf = RandomForestClassifier(n_estimators = 30) # un tipo de clasificador particular

#CLASIFICADOR DE RED NEURONAL
X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)                
clf = MLPClassifier(random_state=1, max_iter=600).fit(X_train, y_train)
clf.predict_proba(X_test[:1])
clf.predict(X_test[:5, :])
clf.score(X_test, y_test)
                  
clf.fit(trainData, trainLabels) # modelo entrenado

correct = 0
predict = []
for (data, label) in zip(testData, testLabels):
    assigned = clf.predict(data.reshape(1,-1))[0]
    print(label, assigned)
    predict.append(assigned)
    if label == assigned:
        correct += 1
print('#', 100 * correct / len(testLabels), '% correct')
print(confusion_matrix(testLabels, predict))
print(classification_report(testLabels, predict))
