#coding=utf8
from neuralnetworktool import *
import random
class Neuron():
	def __init__(self,nInput = 0,studyspeed = 1,thresholdFunc = sigmoidX):
		self.nInput = nInput
		#init the weight in range -1 to 1
		randomweights = [random.uniform(-1, 1) for i in range(nInput)]
		self.weights = randomweights
		self.thresholdFunc = thresholdFunc
		self.studyspeed = studyspeed
		self.inputs = []

	def setInputs(self,inputs):
		self.inputs = inputs

	def getInputs(self):
		return self.inputs

	def getOutput(self,inputs):
		self.inputs = inputs
		outputs = 0
		for i in range(len(self.weights)):
			outputs += self.weights[i] * self.inputs[i]
		return round(self.thresholdFunc(outputs),3)


	def adjustWeights(self,inputs,exceptOutput,realOutput = None):
		if realOutput == None:
			f = self.getOutput(inputs)
		else:
			f = realOutput
		d = exceptOutput
		r = (d - f) * f * (1 - f)
		self.adjustWeightsWithR(r)
		return r

	#R反映出网络输出的平凡差对相应的sigmoid函数的输入中的变化的敏感度
	def adjustWeightsWithR(self,R):
		for i in range(len(self.weights)):
			self.weights[i] += self.studyspeed * R * self.inputs[i]

	def getWeights(self):
		return self.weights

	def setWeights(self,weights):
		self.weights = weights

	def setStudySpeed(self,studyspeed):
		self.studyspeed = studyspeed

	def printInfo(self):
		print self.weights


class NeuralNetWork():
	def __init__(self,nInput,nHide,nOutput,studyspeed):
		self.nInput = nInput
		self.nHide = nHide
		self.nOutput = nOutput
		self.studyspeed = studyspeed
		self.inputNeurons = [0 for i in range(nInput)]
		self.hideNeurons = [
		Neuron(nInput = nInput,studyspeed = studyspeed) for i in range(nHide)]
		self.outputNeurons = [
		Neuron(nInput = nHide,studyspeed = studyspeed) for i in range(nOutput)]


	def backpropagation(self,trainings=[],examples=[]):
		for i in range(len(trainings)):
			self.backpropagationSingle(
				training = trainings[i],
				example = examples[i],
				)

	def backpropagationSingle(self,training,example):
		#d为训练集目标输出
		d = example
		self.inputNeurons = training
		hideoutputs = []
		#对输入层，得到隐藏层输出结果
		for hideNeuron in self.hideNeurons:
			hideoutputs.append(hideNeuron.getOutput(self.inputNeurons))
		outputLayeroutputs = []
		#将隐藏层结果作为输出层的是输入，得到输出层的结果
		for outputNeuron in self.outputNeurons:
			outputLayeroutputs.append(outputNeuron.getOutput(hideoutputs))
		#输出层结果与期望结果做比较，并且反向传播调节
		for i in range(self.nOutput):
			#f为实际输出结果
			f = outputLayeroutputs[i]
			outputR = (d - f) * f * (1 - f)
			#对每个输出层神经元进行调节
			self.outputNeurons[i].adjustWeightsWithR(outputR)
			rw = 0
			for w in self.outputNeurons[i].getWeights():
				rw += outputR * w
			#对隐藏层进行调节
			for j in range(self.nHide):
				hideR = hideoutputs[j] * (1 - hideoutputs[j]) * rw
				#对第j个隐藏层神经元调节
				self.hideNeurons[j].adjustWeightsWithR(hideR)

	def getOutput(self,inputs):
		hideoutputs = []
		#对输入层，得到隐藏层输出结果
		for hideNeuron in self.hideNeurons:
			hideoutputs.append(hideNeuron.getOutput(inputs))
		outputLayeroutputs = []
		#将隐藏层结果作为输出层的是输入，得到输出层的结果
		for outputNeuron in self.outputNeurons:
			outputLayeroutputs.append(outputNeuron.getOutput(hideoutputs))
		return outputLayeroutputs

	def printInfo(self):
		print 'hide'
		for hideNeuron in self.hideNeurons:
			hideNeuron.printInfo()
		print 'output'
		for outputNeuron in self.outputNeurons:
			outputNeuron.printInfo()


# neuralnetworkInstance = NeuralNetWork(3,2,1,1)
# trainings = [
# [1,0,1],
# [0,0,1],
# [0,1,1],
# [1,1,1]
# ]
# examples = [
# 0,1,0,1
# ]
# for i in range(500):
# 	neuralnetworkInstance.backpropagation(trainings,examples)
# 	print neuralnetworkInstance.getOutput([1,0,1])
N1 = Neuron(3,1)
N1.setWeights([2,-2,0])
N2 = Neuron(3,1)
N2.setWeights([1,3,-1])
# print N1.getOutput([1,0,1])
# print N2.getOutput([1,0,1])

N3 = Neuron(3,1)
N3.setWeights([3,-2,-1])
# print N3.getOutput([N1.getOutput([1,0,1]),N2.getOutput([1,0,1]),1])
# print N3.getWeights()
# print N3.adjustWeights([N1.getOutput([1,0,1]),N2.getOutput([1,0,1]),1],0)
# print N3.getInputs()
# print N3.getWeights()


d = 0
hideoutputs = []
#对输入层，得到隐藏层输出结果
hideoutputs.append(N1.getOutput([1,0,1]))
hideoutputs.append(N2.getOutput([1,0,1]))
hideoutputs.append(1)
print 'hideoutputs',hideoutputs

outputLayeroutputs = []
#将隐藏层结果作为输出层的是输入，得到输出层的结果
outputLayeroutputs.append(N3.getOutput(hideoutputs))
#输出层结果与期望结果做比较，并且反向传播调节
#f为实际输出结果
f = outputLayeroutputs[0]
print 'output',f
outputR = (d - f) * f * (1 - f)
#对每个输出层神经元进行调节
print 'outputR',outputR
beforeweights = N3.getWeights()
print 'beforeweights',beforeweights
N3.adjustWeightsWithR(outputR)
print 'afterweights',N3.getWeights()
afterweights = N3.getWeights()
rw = afterweights[0] * outputR
print 'rw',rw
#对隐藏层进行调节
hideR = hideoutputs[0] * (1 - hideoutputs[0]) * rw
#对第j个隐藏层神经元调节
print "N1before",N1.getWeights()
N1.adjustWeightsWithR(hideR)
print "N1before",N1.getWeights()


rw = afterweights[1] * outputR
print 'rw',rw
hideR = hideoutputs[1] * (1 - hideoutputs[1]) * rw
#对第j个隐藏层神经元调节
print "N2before",N2.getWeights()
N2.adjustWeightsWithR(hideR)
print "N2before",N2.getWeights()