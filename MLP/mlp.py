''' Feel free to use numpy for matrix multiplication and
	other neat features.
	You can write some helper functions to
	place some calculations outside the other functions
	if you like to.

	This pre-code is a nice starting point, but you can
	change it to fit your needs.
'''
import numpy as np
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt



class mlp:
	def __init__(self, inputs, targets, nhidden):
		self.beta = 1 		
		self.eta = 0.1   	#learning rate
		self.momentum = 0.0
		"""
		Initialisation
		"""
		self.weight_hidden_lst = np.random.randn(np.shape(inputs)[1],nhidden)
		self.bias_hidden_lst = np.zeros((1,nhidden))
		self.bias_output_lst = np.zeros((1,np.shape(targets)[1]))
		self.bias_hidden_lst += np.random.randn(nhidden)
		self.weight_output_lst = np.random.randn(nhidden,np.shape(targets)[1])
		self.bias_output_lst += np.random.randn(np.shape(targets)[1])







	def earlystopping(self, inputs, targets, valid, validtargets, nhidden, iterations=10):
		valid_error = 9999
		for c in range(iterations):
			

			self.train(inputs, targets, nhidden)
			if c%20 == 0:
				valid_test_temp =self.mk_valid(valid,validtargets,nhidden)
				valid_error_temp = np.sum((validtargets-valid_test_temp)**2.)
				if valid_error_temp<valid_error:
					valid_error=valid_error_temp
				else:
					print '\n number or iterations = ',c
					print 'BEST IS FOUND\n',valid_error

					return 
			if c == iterations-1:
				print 'number of iteration (did not find valid) = ',c
				print 'valid_error=', valid_error


	def train(self, inputs, targets, nhidden):
		
			
		"""
		Training
		n = self.eta
		w_jk = self.weight_output_lst
		v_ij = self.weight_hidden_lst
		h_j = hidden_temp
		a_j = a_hidden
		h_k = output_temp
		y_k = y_output
		d_ok = d_output_error
		d_hj = d_hidden_error
		xi = inputs
		"""


		for h in range(np.shape(inputs)[0]): 	#224 training samples
			"""
			Forward phase
			"""
			hidden_temp = np.zeros(nhidden)
			output_temp = np.zeros(np.shape(targets)[1])
			for i in range(np.shape(inputs)[1]): 	#40 data in each training samples
				for j in range(nhidden):   			#updating each hidden_temp
					hidden_temp[j] += self.weight_hidden_lst[i][j]*inputs[h][i]
			for bias_hidden in range(nhidden):
				hidden_temp[bias_hidden] += self.bias_hidden_lst[0][bias_hidden]

			a_hidden = np.zeros((nhidden,1))
			for a in range(nhidden):
				a_hidden[a] = 1/(1+np.exp(-self.beta*hidden_temp[a]))

			for jj in range(np.shape(a_hidden)[0]): 	#12 (or something else)
					for k in range(np.shape(targets)[1]): 	#to each output
						output_temp[k] += self.weight_output_lst[jj][k]*a_hidden[jj]
			for bias_output in range(np.shape(self.bias_output_lst)[0]):
				output_temp[bias_output] += self.bias_output_lst[0][bias_output]

			y_output = np.zeros(np.shape(targets)[1])
			for y in range(np.shape(targets)[1]):
				y_output[y] = 1/(1+np.exp(-self.beta*output_temp[y]))
			"""
			Backwards phase
			"""
			d_output_error = np.zeros((1,np.shape(targets)[1]))
			d_output_error += (targets[h]-y_output)*y_output*(1-y_output)
			d_hidden_error = a_hidden*(1-a_hidden)*np.dot(self.weight_output_lst,np.transpose(d_output_error))

			self.weight_output_lst += np.transpose(self.eta*np.dot(np.transpose(d_output_error),np.transpose(a_hidden)))
			self.bias_hidden_lst += self.eta*d_hidden_error[0]
			self.bias_output_lst += self.eta*d_output_error[0]



			inputs_temp = np.zeros((1,np.shape(inputs)[1]))
			inputs_temp += inputs[h]
			self.weight_hidden_lst += np.transpose(self.eta*(np.dot(d_hidden_error,inputs_temp)))
		order_change = zip(inputs, targets)
		np.random.shuffle(order_change)
		inputs,targets = map(list,zip(*order_change))





	def mk_valid(self,valid,validtargets,nhidden):
		y_output = np.zeros((np.shape(valid)[0],np.shape(validtargets)[1]))
		for h in range(np.shape(valid)[0]): 	

			hidden_temp = np.zeros(nhidden)
			output_temp = np.zeros(np.shape(validtargets)[1])
			for i in range(np.shape(valid)[1]): 	
				for j in range(nhidden):   			
					hidden_temp[j] += self.weight_hidden_lst[i][j]*valid[h][i]
			for bias_hidden in range(nhidden):
				hidden_temp[bias_hidden] += self.bias_hidden_lst[0][bias_hidden]

			a_hidden = np.zeros((nhidden,1))
			for a in range(nhidden):
				a_hidden[a] = 1/(1+np.exp(-self.beta*hidden_temp[a]))

			for jj in range(np.shape(a_hidden)[0]):
					for k in range(np.shape(validtargets)[1]):
						output_temp[k] += self.weight_output_lst[jj][k]*a_hidden[jj]
			for bias_output in range(np.shape(self.bias_output_lst)[0]):
				output_temp[bias_output] += self.bias_output_lst[0][bias_output]

			
			for y in range(np.shape(validtargets)[1]):
				y_output[h][y] = 1/(1+np.exp(-self.beta*output_temp[y]))
		return y_output


	def confusion(self, inputs, targets,nhidden):
		y_output = np.zeros((np.shape(inputs)[0],np.shape(targets)[1]))
		for h in range(np.shape(inputs)[0]): 	#224 training samples

			hidden_temp = np.zeros(nhidden)
			output_temp = np.zeros(np.shape(targets)[1])
			for i in range(np.shape(inputs)[1]): 	#40 data in each training samples
				for j in range(nhidden):   			#updating each hidden_temp
					hidden_temp[j] += self.weight_hidden_lst[i][j]*inputs[h][i]
			for bias_hidden in range(nhidden):
				hidden_temp[bias_hidden] += self.bias_hidden_lst[0][bias_hidden]

			a_hidden = np.zeros((nhidden,1))
			for a in range(nhidden):
				a_hidden[a] = 1/(1+np.exp(-self.beta*hidden_temp[a]))

			for jj in range(np.shape(a_hidden)[0]): 	#12 (or something else)
					for k in range(np.shape(targets)[1]): 	#to each output
						output_temp[k] += self.weight_output_lst[jj][k]*a_hidden[jj]
			for bias_output in range(np.shape(self.bias_output_lst)[0]):
				output_temp[bias_output] += self.bias_output_lst[0][bias_output]


			for y in range(np.shape(targets)[1]):
				y_output[h][y] = 1/(1+np.exp(-self.beta*output_temp[y]))
		output_max = np.zeros(np.shape(targets)[0])
		validtargets_max = np.zeros(np.shape(targets)[0])
		for l in range(np.shape(targets)[0]):

			output_max[l] = np.argmax(y_output[l][:])
			validtargets_max[l] = np.argmax(targets[l][:])

		
		confusion_mtrx = ConfusionMatrix(validtargets_max, output_max)
		print confusion_mtrx
		confusion_mtrx.print_stats()
		confusion_mtrx.plot(backend='seaborn')
		plt.show()







