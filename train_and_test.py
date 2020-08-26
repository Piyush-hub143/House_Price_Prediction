import numpy as np
from data import *
from regression import *



if __name__ == "__main__":

	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	# Training
	_lambda = 12.4
	W = ridge_grad_descent(trainX, trainY, _lambda)
	
	# Prediction on test data
	predY = testX @ W
	print("Actual Price\t\tPredicted Price")
	for i in range(len(testY)):
		print("{}\t\t{}".format(testY[i][0], predY[i][0]))
	print()

	# SSE
	testSSE = sse(testX, testY, W)
	print("SSE on Test Data = {}".format(testSSE))