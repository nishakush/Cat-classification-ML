# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:11:49 2020

@author: nisha kushwah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import h5py
from PIL import Image
from scipy import ndimage
import imageio
    
    
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_y_orig = np.array(train_dataset["train_set_y"][:]) 

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])
    
    train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))
    
    return train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()


#example of image
i=1
#plt.imshow(train_x_orig[i])
#print(str(train_y[0,i])+" "+classes[np.squeeze(train_y[:,i])].decode("utf-8"))

#train_x_orig.shape -> (209, 64, 64, 3)

m_train, m_test, num_px = train_x_orig.shape[0], test_x_orig.shape[0], train_x_orig.shape[2]


# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(m_train, -1).T #train_x_flatten.shape ->(12288, 209)
test_x_flatten = test_x_orig.reshape(m_test, -1).T  #test_x_flatten.shape ->(12288, 50)

#Standardizing the datasets
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


def sigmoid(z):
    
    s = 1/(1+np.exp(-z))
    return s



#------
def init_zeros(dim):
    
    
    w = np.zeros([dim,1])
    b = 0

    assert(w.shape == (dim, 1)) #if condition is not right, gives error
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def propagate(w,b,X,Y):
    
    m = X.shape[1]
    
    #forward propagation
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m)*(np.dot(Y,np.log(A).T)+(np.dot((1-Y),np.log(1-A).T)))
    
    #backword propagation
    dw = (1/m)*(np.dot(X,(A-Y).T))
    db = (1/m)*(np.sum(A-Y))
    
    
    cost = np.squeeze(cost)
    
    grads = {"dw":dw, "db":db}
    
    return grads,cost



#-------optimization---------------

def optimize(w,b,X,Y,num_iteration,learning_rate,print_cost=False):
    
    costs=[]
    for i in range(num_iteration):
        
        grads,cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if i%100 ==0:
            costs.append(cost)
        if print_cost and i%100 ==0:
            print("%i-th cost: %f"%(i,cost))
            
           
            
    parameters = {"w":w, "b":b}    
        
    grads = {"dw":dw, "db":db}
    
    return parameters, grads , costs


#--------prediction--------
    
def predict(w,b,X):
    
    m=X.shape[1]
    y_predicted=np.zeros((1,m))
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        if A[0][i]>0.5:
            y_predicted[0][i]=1
        else:
            y_predicted[0][i]=0
            
    assert(y_predicted.shape == (1, m)) 
    return y_predicted        
   

#----model----
    
def model(X_train, Y_train, X_test, Y_test, num_iterations = 1000, learning_rate = 0.005, print_cost = False):
    
    #initialize parameters
    w, b = init_zeros(X_train.shape[0])
    
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost=True)
    
    w = parameters["w"]
    b = parameters["b"]
    
    y_predicted_test = predict(w,b,X_test)
    y_predicted_train = predict(w,b,X_train)
    
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_predicted_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predicted_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": y_predicted_test, 
         "Y_prediction_train" : y_predicted_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


d = model(train_x, train_y, test_x, test_y, num_iterations = 2500, learning_rate = 0.005, print_cost = True)



costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

#-------random image testing-----
my_image = "tom.jpg"   


# We preprocess the image to fit your algorithm.
fname = "image/" + my_image

image= np.array(imageio.imread(fname, pilmode='RGB'))
print(image.shape)

my_image = np.array(Image.fromarray((image).astype(np.uint8)).resize((num_px, num_px)).convert('RGB')).reshape((1, num_px*num_px*3)).T

my_image = my_image/255.
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")




























