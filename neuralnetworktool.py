#coding=utf8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def matrix(arg):
	return numpy.matrix(arg)

# 	x:The independent variable
#	k:Steep degree of sigmoid function
def sigmoidX(x,k=1):
	return 1 / ( 1 + np.exp(-x*k))
#get linspace output of sigmoidX
def sigmoid(x_list):
	ys = []
	for x in x_list:
		ys.append(sigmoidX(x))
	return ys



