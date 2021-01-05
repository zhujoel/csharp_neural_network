
       **** Description of the projects in the NeuralNetwork solution ****




NeuralNetwork.Common (.NET Standard class library)

Description: 
  This class library contains the definitions and interfaces that will be common to all the implementations. 
  THIS PROJECT IS MEANT TO BE READONLY! It must not be modified.
  
_________________________________________________________________
  
NeuralNetwork (.NET Standard class library)

Description:
  This class library is the one that will contain the implementation of the neural network, its layers, etc.
  Everything in this project can be modified as much as necessary, as long as the classes implement the interfaces defined in NeuralNetwork.Common.

_________________________________________________________________  

BooleanFunctionTester (.NET Framework WPF application)

Description:
  This application permits to load a serialized network with an input layer of size 2 and an output layer of size 1. The network is meant to represent a binary boolean function. The user can then input a first and a second argument (0 or 1 each time) and check the output of the network.
  This application can be used once forward propagation has been implemented; three serialized networks are available.

_________________________________________________________________ 

PropagationComparison project (.NET Framework console application)

Description:
  This program loads a serialized network, performs a forward propagation on a series of inputs and stores this first set of results.
  Afterwards it updates the weights of the network using a series of gradients provided by the user, performs a second forward propagation on the same series of inputs and stores this second set of results.
  Both forward propagation results are then output in json format, either in a file or on the console.

Parameters: 
  - The path to a serialized network
     Batch size 1
	 Input size: N
	 Output size: 1
  - The path to the inputs of the network, in csv format
     Number of inputs: P
	 Size of each input (line in the csv file): N 
  - The path to the gradients of the network output, in csv format
     Number of gradients: P
	 Size of each gradient (line in the csv file): 1
  - (Optional): The path to the file in which the results will be stored (json format). Otherwise they are output to the console.
  
_________________________________________________________________ 
 
RegressionConsole project (.NET Framework console application)

Description:
  This program loads a serialized network, runs it on the test data from the PricingDataProvider and summarizes the errors between the network outputs and the expected outputs.
  The error summary is output in json format, either in a file or on the console.
  
  Parameters: 
  - The path to a serialized network
  - (Optional): The path to the file in which the summary will be stored (json format). Otherwise it is output to the console.