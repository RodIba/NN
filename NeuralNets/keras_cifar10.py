from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD 
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse 

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, 
	help = "path to the output loss/accuracy plot")

args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data ...")

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", 
	"frog", "horse", "ship", "truck"]

model = Sequential()
model.add(Dense(1024, input_shape = (3072, ), activation = "relu"))
model.add(Dense(512, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])
model.fit(trainX, trainY, validation_data = (testX, testY),
	epochs = 100, batch_size = 32)

print("[INFO] evaluating network... ")
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1),
		 target_names = labelNames))



