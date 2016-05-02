# NeuralNetwork
利用Python实现的BP神经网络进行人脸识别

实现的BP神经网络为典型的三层神经网络

#定义
    #nInput : Number of Neuron in input layer
    #nHide : Number of Neuron in hide layer
    #nOutput : Number of Neuron in output layer
    #studyspeed ： Learning Rate
    network = NeuralNetWork(nInput = nInput , nHide = nHide,nOutput = nOutput,studyspeed = 0.05)

#训练过程
    #Training with single example
    #training : [input0,input1,input2...]
    #example : [output0,output1,output2...] (excepted output)
    network.backpropagationSingle(training = training,example = example)
				
#使用
    #Get the output of network when the input is inputTest
		res =  network.getOutput(inputTest)
				
