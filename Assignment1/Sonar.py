# imports

import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
import math
import pandas as pd
import sys
import pickle
import os 


# activation functions

def perceptron(z):
    return -1 if z<=0 else 1

# loss functions

def ploss(yhat, y):
    return max(0, -yhat*y)

# prediction functions

def ppredict(self, x):
    return self(x)




class Sonar_Model:
    
    
    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x), predict=ppredict):
    
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)
    
    def __str__(self):
        
        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
    
     #Sonar class should have a predict(v) method that uses internal weights to make prediction on new data 
    def ___call__(self, v):
        
        yhat = self._a(np.dot(self.w, np.array(v)) + self.b)
        return yhat

    
    def load_model(self, file_path):
        '''
        open the pickle file and update the model's parameters
        '''
        return pickle.load(open(file_path,'rb'))

    def save_model(self):
        '''
        save your model as 'sonar_model.pkl' in the local path
        '''
        
        sonar_model = self
        
        return pickle.dump(sonar_model, open(file_path,'wb') )
        
        
        
        
class Sonar_trainer:
    
    def __init__(self, dataset, model):
        
        self.dataset = dataset
        self.model = model
        self.loss = ploss

    def accuracy(self, data):
        '''
        return the accuracy on data given data iterator
        '''
        acc = 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])
        return acc
    
    
    
  #Sonar class should have a public method train, which trains the perceptron on loaded data
    def train(self, lr, ne):
        '''
        This method should:
        1. display initial accuracy on the training data loaded in the constructor
        2. update parameters of the model instance in a loop for ne epochs using lr learning rate
        3. display final accuracy
        '''
        
        print("training model on data...")
        
        for epoch in range(ne):
            for d in self.dataset:
                x, y = d
                x = np.array(x)
                yhat = self.model(x)
                error = y - yhat
                self.model.w += lr*(y-yhat)*x
                
                #internal weights property: list of floats representing perceptron weights, updated by the train() method
                internal_w = self.model.w
                
                self.model.b += lr*(y-yhat)
            accuracy = self.accuracy(self.dataset)    
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))
            
        print("training complete")
        
        #Train method returns a single float representing the mean squarre error on the trained set
        print("final mean square error: %.3f" % (accuracy))
        
        
        
        
class Sonar_Data:


#Sonar Class should have the datafile relative path (string) and name (string) as contrustor arguments
        
    def __init__(self, relative_path='C:/Users/gigi-/OneDrive/Documents/MA2/GitHub/ProjetAI/IntroAI/keio2019aia/data/assignment1', data_file_name='sonar_data.pkl'):
        '''
        initialize self.index; load and preprocess data; shuffle the iterator
        '''
        self.index = -1
        self.data = list()
        self.full = list()
        full_path = os.path.join(relative_path,data_file_name)
        Sonar_Raw = pickle.load(open(full_path,'rb'))
       
        
    def __iter__(self):
        '''
        See example code (ngram) in lecture slides
        '''
        return self

    def __next__(self):
        '''
        See example code (ngram) in slides
        '''
        self.index += 1
        self.full += self.data
        
        if self.index <= len(Sonar_Raw['r']):
            self.data = [(Sonar_Raw['r'][self.index], -1)]+[(Sonar_Raw['m'][self.index], 1)]
        else: 
            self.data = [(Sonar_Raw['m'][self.index], 1)]

  
        if self.index == len(Sonar_Raw):
            raise StopIteration
        
        return self.full

    def _shuffle(self):
        '''
        shuffle the data iterator
        '''
        return random.shuffle(self)
        
        
        
        
        
        
def main():

    data = Sonar_Data()
    model = Sonar_Model(dimension=60, activation=perceptron)  # specify the necessary arguments
    trainer = Sonar_Trainer(data, model)
    trainer.train() # experiment with learning rate and number of epochs
    model.save_model()


if __name__ == '__main__':

    main()      
