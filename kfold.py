import numpy as np
import matplotlib.pyplot as plt
from data import *
from regression import *



def k_fold_cross_validation(X, Y, k, lambdas):
	# Input:
	# 	X: [N x D] (features)
	# 	Y: [N x 1] (prices)
	# 	k: number of splits to perform while doing kfold cross validation
	#	lambdas: list of parameter lambda to try
	# Output:
	# 	SSEs: List of avg SSE over each split for each given lambda
	
	n, d = X.shape
	size = int(n / k)
	SSEs = []

	for _lambda in lambdas:
		print("Running for lambda = {}".format(_lambda))
		SSE = 0.0
		for i in range(k):
			start = i*size
			end = (i+1)*size
			Xval = X[start:end]
			Yval = Y[start:end]
			Xtrain = np.append(X[0:start], X[end:], axis=0)
			Ytrain = np.append(Y[0:start], Y[end:], axis=0)
			W = ridge_grad_descent(Xtrain, Ytrain, _lambda)
			sse1 = sse(Xval, Yval, W)
			SSE += sse1
			print("\tSSE for split {}/{} = {}".format(i+1, k, sse1))
		SSEs.append(SSE/k)
		print("\tAvg SSE = {}".format(SSE/k))
	return SSEs



def plot_kfold(lambdas, sses):
	# Input:
	# 	lambdas: list of parameter lambda
	# 	sses: list of average SSE values for each lambda
	# Output:
	# 	Plots SSE vs lambda graph
	plt.plot(lambdas, sses)
	plt.ylabel('Validation_SSE')
	plt.xlabel('Lambda')
	plt.show()




if __name__ == "__main__":

	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = [12.0,12.1,12.2,12.3,12.4,12.5,12.6,12.7,12.8,12.9,13.0]
	sses = k_fold_cross_validation(trainX, trainY, 6, lambdas)

	minSSE = min(sses)
	idx = sses.index(minSSE)
	print("Minimum Avg SSE = {}, for lambda = {}".format(minSSE, lambdas[idx]))
	plot_kfold(lambdas, sses)
	



