import pandas as pd
import tensorflow
from tensorflow import keras
from keras import layers
import numpy as np
from termcolor import colored


def vectorize(data):
    array = np.zeros(shape=(len(data), len(data.columns)))
    for count, series in data.iterrows():
        for i in range(len(series)):
            array[count][i] = series[i]
    return array


test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")

labels = train_data.iloc[:, 1]
labels = pd.get_dummies(labels)
labels = vectorize(labels)

train_data = train_data.drop(["PassengerId"], axis=1)
train_data = train_data.drop(["Survived"], axis=1)
train_data = train_data.drop(["Name"], axis=1)

features = pd.get_dummies(train_data)
features = vectorize(features)

# for row_number in range(len(features)):
#     for i in features[row_number]:
#         if i == np.nan:
#             features[row_number][i] == 0

print(features)
# features = vectorize(train_data)
# features_val = features[:int(len(features) / 2)]
# partial_features = features[int(len(features) / 2):]
# labels_val = labels[:int(len(labels) / 2)]
# partial_labels = labels[int(len(labels) / 2):]

model = keras.Sequential([
    layers.Dense(16, "relu"),
    layers.Dense(16, "relu"),
    layers.Dense(16, "relu"),
    layers.Dense(1, "sigmoid")
])

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"]
              )
model.fit(features,
          labels,
          epochs=20,
          batch_size=512,
          )
