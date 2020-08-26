import numpy as np



def sse(X, Y, W):
	# Input:
	# 	X: [N x D] (features)
	# 	Y: [N x 1] (prices)
	# 	W: [D x 1] (weights)
	# Output:
	# 	Sum Squared Error
	return np.linalg.norm(Y - X @ W) ** 2



def ridge_objective(X, Y, W, _lambda):
	# Input:
	# 	X: [N x D] (features)
	# 	Y: [N x 1] (prices)
	# 	W: [D x 1] (weights)
	# 	_lambda: parameter
	# Output:
	# 	Ridge Objective Function Value (Loss)
	return sse(X,Y,W) + _lambda * np.linalg.norm(W, ord=2)




def grad_ridge(W, X, Y, _lambda):
	# Input:
	# 	W: [D x 1] (weights)
	# 	X: [N x D] (features)
	# 	Y: [N x 1] (prices)
	# 	_lambda: parameter
	# Output:
	# 	Gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	return -2 * (X.T @ (Y - X @ W)) + 2 * _lambda * W




def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	# Input:
	# 	X: [N x D] (features)
	# 	Y: [N x 1] (prices)
	# 	_lambda: parameter
	# 	max_iter: maximum number of iterations of gradient descent to run in case of no convergence
	# 	lr : learning rate
	# 	epsilon: gradient norm below which we can say that the algorithm has converged 
	# Output:
	# 	W: Trained weight vector [D X 1]

	n, d = X.shape
	W = np.ones([d,1])
	for _ in range(max_iter):
		grad = grad_ridge(W, X, Y, _lambda)
		if np.linalg.norm(grad) < epsilon:
			break
		W = W - lr * grad
	return W


