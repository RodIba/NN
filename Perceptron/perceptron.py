import numpy as np


class Perceptron:
	def __init__(self,N, alpha = 0.1):
	#initialize weigth matrix and store the learning rate
		self.W = np.random.randn(N+1)/np.sqrt(N)
		self.alpha = alpha 
	
	def step(self, x):
	#apply the stop function
		return 1 if x > 0 else 0

	def fit(self, X, y, epochs = 20):
		#insert a column of 1's as the last entry in the feature 
		#matrix -- this little trick allows us to treat the bias 
		# as the trainable parameter within the weight matrix 

		X = np.c_[X, np.ones((X.shape[0]))]

		# Loop over the desired number of epochs 
		
		for epoch in np.arange(0, epochs):
			#Loop over each individual data point 
			for (x, target) in zip(X, y): # y target output class labels 
			# take the product between the input features 
			# and the weight matrix, then pass this value 
			# through the step function to obtai the prediction 
				p = self.step(np.dot(x, self.W))

			# only perform a weight update if our prediction 
			# does not match the target 
			if p != target:
				#determine the error 
				error = p - target 

				#update the weight matrix 
				self.W += -self.alpha * error * x

	def predict(self, X, addBias=True):
		# ensure out input is a matrix 
		X = np.atleast_2d(X)

		# check if bias column should be added 

		if addBias:
			# insert a column of 1's as the last entry in the feature 
			# matrix (bias)
		        X = np.c_[X, np.ones((X.shape[0]))]
		#take the dot product 
		return self.step(np.dot(X, self.W))
