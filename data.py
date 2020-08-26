import numpy as np
import pandas as pd


def read_data(filename):
	# Input: filename
	# Output: 
	# 	X: [N x D] (features)   (N: number of datapoints, D: number of feature dimensions)
	# 	Y: [N x 1] (prices)
	dataframe = pd.read_csv(filename, keep_default_na=False, na_values='')
	data = [np.array(dataframe[col]) for col in dataframe]
	for i, d in enumerate(data):
		data[i].shape = (data[i].shape[0], 1)
	data = np.concatenate(data, axis = 1)
	X = data[:,:-1]
	Y = data[:,-1]
	Y.shape = (Y.shape[0],1)
	return X, Y


def one_hot_encode(X, labels):
	# Input:
	# 	X: [N X 1] 
	# 	labels: list of all possible labels for current feature
	# Output:
	# 	newX: [N X len(labels)] in one hot encoded format
	
	X.shape = (X.shape[0], 1)
	newX = np.zeros((X.shape[0], len(labels)))
	label_encoding = {}
	for i, l in enumerate(labels):
		label_encoding[l] = i
	for i in range(X.shape[0]):
		newX[i, label_encoding[X[i,0]]] = 1
	return newX


def preprocess(X, Y):
	# Input:
	# 	X: [N x D] 
	# 	Y: [N x 1]
	# Output:
	# 	ansX: [N x D] (normalised and one hot encoded, increased dimensionality)
	# 	ansY: [N x 1] (same as input)
	
	def normalise(l):
		if isinstance(l[0], str):
			labels = np.unique(l)
			return one_hot_encode(l, labels)
		else:
			mu = np.mean(l)
			sigma = np.std(l)
			return ((l - mu)/sigma).reshape(len(l),1)

	n, d = X.shape
	# Appended ones to act as bias
	ansX = np.ones([n,1])
	ansY = np.copy(Y)
	# Ignored first column as it was serial number
	for i in range(1,d):
		ansX = np.append(ansX, normalise(X[:,i]), axis=1)

	return ansX.astype(float), ansY.astype(float)



def separate_data(X, Y):
	# Input:
	# 	X: [N x D] 
	# 	Y: [N x 1]
	# Output:
	# 	trainX: [1200 x D] (first 1200 as training data)
	# 	trainY: [1200 x 1]
	# 	testX: [N-1200 x D] (remaining as test data)
	# 	testY: [N-1200 x 1]

	trainX = X[0:1200, :]
	trainY = Y[0:1200, :]
	
	testX = X[1200:, :]
	testY = Y[1200:, :]

	return trainX, trainY, testX, testY


