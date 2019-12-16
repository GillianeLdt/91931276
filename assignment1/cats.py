#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbconvert --to script Cats.ipynb')


# In[1]:


import numpy as np
import pickle as pkl
import random
import os


# In[2]:


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lrloss(yhat, y):
    return 0.0 if yhat==y else -1.0*(y*np.log(yhat)+(1-y)*np.log(1-yhat))


# In[37]:


class Cat_Model:

    def __init__(self, dimension=None, weights=None, bias=None, activation=(lambda x: x), predict=lrpredict):

        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)*np.sqrt(1/dimension)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        #self.predict = predict.__get__(self)

    def __str__(self):
        '''
        display the model's information
        '''
        info = "Simple cell neuron\n        \tInput dimension: %d\n        \tBias: %f\n        \tWeights: %s\n        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
        return info

    def __call__(self, x):
        '''
        return the output of the network
        '''
        #probability to belong to a cat 
        yhat = self._a(np.dot(self.w, np.reshape(x, x.size))+self.b)
        return yhat
    
    def lrpredict(self, x):
        return 1 if self(x)>0.5 else 0

    def load_model(self, file_path):
        '''
        open the pickle file and update the model's parameters
        '''
        file = pkl.load(open(file_path,'rb'))
        
        self._dim = file._dim
        self.w = file.w
        self.b = file.b
        self._a = file._a


    def save_model(self):
        '''
        save your model as 'cat_model.pkl' in the local path
        '''
        saved = open('cat_model.pkl','wb')
        cat_model = pkl.dump(self, saved)
        saved.close


# In[34]:


class Cat_Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.loss = lrloss

    def accuracy(self, data):
        '''
        return the accuracy on data given data iterator
        '''
        acc = 100*np.mean([1 if self.model.lrpredict(x) == y else 0 for x, y in data])
        return acc

    def train(self, lr, ne):
        '''
        This method should:
        1. display initial accuracy on the training data loaded in the constructor
        2. update parameters of the model instance in a loop for ne epochs using lr learning rate
        3. display final accuracy
        '''
        
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))
        
        costs=[]
        accuracies=[]
        
        for epoch in range(ne):
            J=0
            dw=0
            
            self.dataset._shuffle()
            for d in self.dataset:
                xi, yi = d
                #x = np.array(x)
                yhat = self.model(xi)
                print("y: "+str(yi)+", \t yhat: "+str(yhat)+", \t self.dataset.index: "+str(self.dataset.index))
                J += self.loss(yhat, yi)
                dy = yhat - yi
                dw += xi*dy
                
            J /= len(self.dataset.simple)
            dw /= len(self.dataset.simple)
            self.model.w= self.model.w - lr*dw
       
            accuracy = self.accuracy(self.dataset)
            print('>epoch=%d, learning_rate=%.3f, accuracy=%.3f' % (epoch+1, lr, accuracy))
            costs.append(J)
            accuracies.append(accuracy)
            
        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))
        


# In[35]:


class Cat_Data:

    def __init__(self, relative_path='C:/Users/gigi-/keio2019aia/data/assignment1', data_file_name='cat_data.pkl'):
        '''
        initialize self.index; load and preprocess data; shuffle the iterator
        '''
        
        self.index = -1
        
        self.relative_path = relative_path
        self.data_file_name=data_file_name
        
        full_path = os.path.join(relative_path,data_file_name)
        raw = pkl.load(open(full_path,'rb'))
        self.simple = [(np.reshape(d, d.size), 1) for d in self.standardize(raw['train']['cat'])]+[(np.reshape(d, d.size), 0) for d in self.standardize(raw['train']['no_cat'])]
        
    def standardize (self, rgb_images):
        mean = np.mean(rgb_images, axis=(1,2), keepdims=True)
        std = np.std(rgb_images, axis=(1,2), keepdims=True)
        
        #standardized_cat=standardize(cat_data['train']['cat'])[0]
        return (rgb_images - mean) / std

    
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
  
        if self.index == len(self.simple):
            self.index = -1
            raise StopIteration
        
        return self.simple[self.index][0], self.simple[self.index][1]

    def _shuffle(self):
        '''
        shuffle the data iterator
        '''
        return random.shuffle(self.simple)


# In[51]:


def main():

    data = Cat_Data()
    model = Cat_Model(dimension = (64*64*3), activation=sigmoid)  # specify the necessary arguments
    trainer = Cat_Trainer(data, model)
    trainer.train(0.01, 85) # experiment with learning rate and number of epochs
    model.save_model()


if __name__ == '__main__':

    main()

