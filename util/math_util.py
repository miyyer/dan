from numpy import *

def softmax(w):
    ew = exp(w - max(w))
    return ew / sum(ew)

def relu(x):
    return x * (x > 0)

def drelu(x):
    return x > 0
    
def crossent(label, classification):
    return -sum(label * log(classification))

def dcrossent(label, classification):
    return classification - label