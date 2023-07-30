from random import random
import math

class Neuron:
    def __init__(self, weights, bias):
        # self.act = act
        self.weights = weights
        self.bias = bias
        self.inputs = None
        self.output = 0
        self.act_output = 0
        # self.error = None
        # self.act_error = None
        self.dw = [0 for i in range(len(weights))]
        self.db = 0

class Layer:
    def __init__(self, prevL, num, act, dact, weights, biases):
        self.prevE = None
        self.E = None
        self.prevL = prevL
        self.num = num
        self.act = act
        self.dact = dact
        self.neurons = [Neuron(weights[i], biases[i]) for i in range(num)]
    
    def feedforward(self, inputs):
        for i in range(self.num):
            self.neurons[i].inputs = inputs
            self.neurons[i].output = sum([j*k for j,k in zip(self.neurons[i].inputs,self.neurons[i].weights)]) + self.neurons[i].bias
            self.neurons[i].act_output = self.act(self.neurons[i].output)
        return [self.neurons[i].act_output for i in range(self.num)]
    
    def backprop(self, errors):
        self.E = errors
        self.prevE = [0 for _ in range(self.prevL)]
        for i in range(self.prevL):
            self.prevE[i] = sum([errors[p]*self.dact(self.neurons[p].output)*self.neurons[p].weights[i] for p in range(self.num)])

        for i in range(self.num):
            for j in range(len(self.neurons[i].weights)):
                self.neurons[i].dw[j] = self.E[i]*self.dact(self.neurons[i].output)*self.neurons[i].inputs[j]
            self.neurons[i].db = self.E[i]*self.dact(self.neurons[i].output)

        return self.prevE
    
class ANN:
    def __init__(self, arch, acts, dacts, Lweights, Lbiases):
        self.arch = arch
        self.acts = acts
        self.layers = [Layer(arch[i-1], arch[i], acts[i], dacts[i], Lweights[i], Lbiases[i]) for i in range(1,len(arch))]

    def feedforward(self, inputs):
        for i in range(len(self.layers)):
            inputs = self.layers[i].feedforward(inputs)
        return inputs
    
    def backprop(self, errors):
        for i in range(len(self.layers)-1,-1,-1):
            errors = self.layers[i].backprop(errors)
        return errors
    
    def update(self, lr):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                for k in range(len(self.layers[i].neurons[j].weights)):
                    self.layers[i].neurons[j].weights[k] -= lr * self.layers[i].neurons[j].dw[k]
                self.layers[i].neurons[j].bias -= lr * self.layers[i].neurons[j].db

    def zero_grad(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                for k in range(len(self.layers[i].neurons[j].weights)):
                    self.layers[i].neurons[j].dw[k] = 0
                self.layers[i].neurons[j].db = 0
    
    def display(self):
        for i in range(len(self.layers)):
            print(f"######### Layer {i} ##############")
            for j in range(len(self.layers[i].neurons)):
                print(f'Neuron{i+1}/{j+1}')
                print(f'Weights/B:{self.layers[i].neurons[j].weights}+{self.layers[i].neurons[j].bias}')
                print(f"I/O/Act_O:{self.layers[i].neurons[j].inputs}, {self.layers[i].neurons[j].output}, {self.layers[i].neurons[j].act_output}")
                print(f"Error/Prev_Err:{self.layers[i].E}, {self.layers[i].prevE}")  
                print(f'dw/db:{self.layers[i].neurons[j].dw}/{self.layers[i].neurons[j].db}')
                print()
            print()
        pass

    def train(self, inputs, targets, epochs=1, lr=0.1):
        for i in range(epochs):
            print(f'Epoch {i+1}')
            for j in range(len(inputs)):
                outputs = self.feedforward(inputs[j])
                print("##########Feedforward##########")
                self.display()
                errors = [outputs[k]-targets[j][k] for k in range(len(outputs))]
                print("##########Error##########")
                print(errors)
                self.backprop(errors)
                print("##########Backprop##########")
                self.display()
                self.update(lr)
                print("##########Update##########")
                self.display()
                self.zero_grad()
                print("##########Zero Grad##########")
                self.display()
                print()

def sigmoid(x):
    return 1/(1+math.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def randomN():
    return random()-0.5

def randomList(n):
    return [randomN() for _ in range(n)]

def randomWeights(l):
    return [None]+[[randomList(l[i-1]) for _ in range(l[i])] for i in range(1,len(l))]

def randomBiases(l):
    return [None]+[randomList(l[i]) for i in range(1,len(l))]

# nn=ANN([2, 3, 1],[sigmoid,sigmoid,sigmoid],[dsigmoid,dsigmoid,dsigmoid],[[[0.1,0.4],[0.2,0.5],[0.3,0.6]],[[0.1,0.4],[0.2,0.5],[0.3,0.6]],[[0.2,0.3,0.4]]],[[0.1,0.2,0.3],[0.1,0.2,0.3],[0.1]])
arch = [2, 2, 1]
print(randomWeights(arch))
print(randomBiases(arch))
nn=ANN(arch,[None,sigmoid,sigmoid],[None,dsigmoid,dsigmoid],randomWeights(arch),randomBiases(arch))
nn.display()
# nn.train([[0.1,0.5]],[[0.8]],epochs=1,lr=0.1)
nn.train([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]],[[0.0],[1.0],[1.0],[0.0]],epochs=10000,lr=0.1)
# nn.train([[1.0,1.0]],[[0.0]],epochs=1,lr=0.1)
# nn.train([[1.0,1.0]],[[1.0]],epochs=1,lr=1)

print("##########Final##########")
print(f"inputs [0,0]: {nn.feedforward([0.0,0.0])}")
print(f"inputs [0,1]: {nn.feedforward([0.0,1.0])}")
print(f"inputs [1,0]: {nn.feedforward([1.0,0.0])}")
print(f"inputs [1,1]: {nn.feedforward([1.0,1.0])}")