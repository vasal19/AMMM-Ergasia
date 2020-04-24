#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:59:34 2020

@author: vasilis
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def perceptron(x, t, maxEpochs, beta):
    w = np.random.rand(1,5)
    y = np.zeros(len(t))
    u = y
    flag = False
    epoch = 1
    while epoch <= maxEpochs and flag == False:
        flag = True
        for p in range(1, len(x)):
            u[p] = x[p,0] * w[0,0] + x[p,1] * w[0,1] + x[p,2] * w[0,2] + x[p,3] * w[0,3]
            if u[p] < 0:
                y[p] = -1
            else :
                y[p] = 1
            if t[p] != y[p]:
                w[0,1] = w[0,1] + beta * (t[p] - y[p]) * x[p,0]
                w[0,2] = w[0,2] + beta * (t[p] - y[p]) * x[p,1]
                w[0,3] = w[0,3] + beta * (t[p] - y[p]) * x[p,2]
                w[0,4] = w[0,4] + beta * (t[p] - y[p]) * x[p,3]
                flag = False 
        epoch += 1
    return w

def adaline(x, t, maxEpochs, beta, minmse):
    w = np.random.rand(1, 5)
    flag = False
    y = np.zeros(len(t))
    u = y
    epoch = 1
    while epoch <= maxEpochs and flag == False:
        for p in range(1, len(x)):
            u[p] = x[p,0] * w[0,0] + x[p,1] * w[0,1] + x[p,2] * w[0,2] + x[p,3] * w[0,3]
            y[p] = u[p]
            w[0,1] = w[0,1] + beta * (t[p] - y[p]) * x[p,0]
            w[0,2] = w[0,2] + beta * (t[p] - y[p]) * x[p,1]
            w[0,3] = w[0,3] + beta * (t[p] - y[p]) * x[p,2]
            w[0,4] = w[0,4] + beta * (t[p] - y[p]) * x[p,3]
        mse = np.square(np.subtract(t,y)).mean()
        mse = mean_squared_error(t,y) 
        print ("Εποχή" + str(epoch))
        print ("Σφάλμα" + str(mse))
        if mse < minmse:
            flag = True
        epoch += 1
    return w

data = read_csv('iris.data', header=None).values
numberOfPatterns, numberOfAttributes = data.shape

x = data[:, :4]
plt.plot(x[:50,0], x[:50,2], 'b.')
plt.plot(x[50:100,0], x[50:100,2], 'r.')
plt.plot(x[100:150,0], x[100:150,2], 'g.')
plt.show()
plt.clf()

t = np.zeros(numberOfPatterns)

ans = 'y'
while ans == 'y':
    print ("1 Διαχωρισμός Iris-setosa από (Iris-versicolor και Iris-virginica)")
    print ("2 Διαχωρισμός Iris-virginica από (Iris-setosa και Iris-versicolor)")
    print ("3 Διαχωρισμός Iris-versicolor από (Iris-setosa και Iris-virginica)")
    choice = int(input("Επιλέξτε (1/2/3) "))
    if choice == 1:
        map_dict = {
            "Iris-setosa": 1,
            "Iris-versicolor": 0,
            "Iris-virginica": 0
            }
    elif choice == 2:
        map_dict = {
            "Iris-setosa": 0,
            "Iris-versicolor": 0,
            "Iris-virginica": 1
            }
    else:
        map_dict = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 0
            }
    
    for pattern in range(0,numberOfPatterns):
        t[pattern] = map_dict[data[pattern][4]]

    x = np.hstack((x, np.ones((numberOfPatterns, 1), dtype=float)))
    xtrain = np.vstack((x[:40,:], x[50:90,:], x[100:140,:])).astype(float)
    xtest = np.vstack((x[40:50,:], x[90:100,:], x[140:150,:])).astype(float)
    ttrain = np.hstack((t[:40], t[50:90], t[100:140])).astype(float)
    ttest = np.hstack((t[40:50], t[90:100], t[140:150])).astype(float)
    print(ttest)
    plt.plot(xtrain[:,0], xtrain[:,2], 'b.')
    plt.plot(xtest[:,0], xtest[:,2], 'r.')
    plt.show()
    plt.clf()
    
    algoChoice = 0
    while algoChoice != 4:
        print("1. Υλοποίηση με Perceptron")
        print("2. Υλοποίηση με Adaline")
        print("3. Υλοποίηση με Λύση Ελάχιστων Τετραγώνων")
        print("4. Επιστροφή στο αρχικό μενού")
        algoChoice = int(input("Επιλέξτε (1/2/3/4): "))
        
        if algoChoice == 1:
            maxEpochs = int(input("Δώσε τιμή για το μέγιστο αριθμό επαναλήψεων: "))
            beta = float(input("Δώσε τιμή για τον συντελεστή εκπαίδευσης: "))
            
            w = perceptron(xtrain, ttrain, maxEpochs, beta)
            w1 = np.transpose(w)
            yTest = xtest.dot(w1)
            
            predictTest = np.zeros(len(ttest))
            for i in range(0, len(ttest)):
                if yTest[i] < 0:
                    predictTest[i] = 0
                else:
                    predictTest[i] = 1
            
            plt.plot(ttest[:], 'b.')
            plt.plot(predictTest[:], 'r.')
            plt.show()
            plt.clf()
            
            plt.figure(0)
            plt.figure(1)
            for k in range(0, 9):
                xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
                xtrain = xtrain.astype(float)
                xtest = xtest.astype(float)
                ttrain = ttrain.astype(float)
                ttest = ttest.astype(float)
                
                plt.figure(0)
                plt.subplot(3, 3, k+1)
                plt.plot(xtrain[:,0], xtrain[:,2], 'b.')
                plt.plot(xtest[:,0], xtest[:,2], 'r.')

                w = perceptron(xtrain, ttrain, maxEpochs, beta)
                w1 = np.transpose(w)
                yTest = xtest.dot(w1)
                
                predictTest = np.zeros(len(ttest))
                for i in range(0, len(ttest)):
                    if yTest[i] < 0:
                        predictTest[i] = 0
                    else:
                        predictTest[i] = 1
                
                plt.figure(1)
                plt.subplot(3, 3, k+1)
                plt.plot(ttest[:], 'b.')
                plt.plot(predictTest[:], 'r.')
            plt.show()
            plt.clf()
            
        elif algoChoice == 2:
            ttrain1 = np.zeros(len(ttrain))
            ttest1 = np.zeros(len(ttest))
            
            for pattern in range(0, len(xtrain)):
                if ttrain[pattern] == 1:
                    ttrain1[pattern] = 1
                else:
                    ttrain1[pattern] = -1
            for pattern in range(0, len(ttest)):
                if ttest[pattern] == 1:
                    ttest1[pattern] = 1
                else:
                    ttest1[pattern] = -1
            print(ttest)
            maxEpochs = int(input("Δώσε τιμή για το μέγιστο αριθμό επαναλήψεων: "))
            beta = float(input("Δώσε τιμή για τον συντελεστή εκπαίδευσης: "))
            minmse = float(input("Δώσε τιμή για το ελάχιστο σφάλμα: "))
            print("Για ένα fold.")
            w = adaline(xtrain, ttrain1, maxEpochs, beta, minmse)
            w1 = np.transpose(w)
            yTest = xtest.dot(w1)
            
            predictTest = np.zeros(len(ttest1))
            for i in range(0, len(ttest1)):
                if yTest[i] < 0:
                    predictTest[i] = 0
                else:
                    predictTest[i] = 1
            
            plt.plot(ttest[:], 'b.')
            plt.plot(predictTest[:], 'r.')
            plt.show()
            plt.clf()
            
            plt.figure(0)
            plt.figure(1)
            for k in range(0, 9):
                xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
                xtrain = xtrain.astype(float)
                xtest = xtest.astype(float)
                ttrain = ttrain.astype(float)
                ttest = ttest.astype(float)
                
                ttrain1 = np.zeros(len(ttrain))
                ttest1 = np.zeros(len(ttest))
                
                for pattern in range(0, len(ttrain)):
                    if ttrain[pattern] == 1:
                        ttrain1[pattern] = 1
                    else:
                        ttrain1[pattern] = -1
                for pattern in range(0, len(ttest)):
                    if ttest[pattern] == 1:
                        ttest1[pattern] = 1
                    else:
                        ttest1[pattern] = -1
                
                plt.figure(0)
                plt.subplot(3, 3, k+1)
                plt.plot(xtrain[:,0], xtrain[:,2], 'b.')
                plt.plot(xtest[:,0], xtest[:,2], 'r.')
                
                w = adaline(xtrain, ttrain1, maxEpochs, beta, minmse)
                w1 = np.transpose(w)
                yTest = xtest.dot(w1)
                print (ttest)
                predictTest = np.zeros(len(ttest))
                for i in range(0, len(ttest1)):
                    if yTest[i] < 0:
                        predictTest[i] = 0
                    else:
                        predictTest[i] = 1
                
                plt.figure(1)
                plt.subplot(3, 3, k+1)
                plt.plot(ttest[:], 'b.')
                plt.plot(predictTest[:], 'r.')
            plt.show()
            plt.clf()     
        elif algoChoice == 3:
            ttrain1 = ttrain
            ttest1 = ttest
            
            for pattern in range(0, len(xtrain)):
                if ttrain[pattern] == 1:
                    ttrain1[pattern] = 1
                else:
                    ttrain1[pattern] = -1
            for pattern in range(0, len(ttest)):
                if ttest[pattern] == 1:
                    ttest1[pattern] = 1
                else:
                    ttest1[pattern] = -1    
            w = np.random.rand(1, 5)
            # TODO Fix w

            w = ttrain1 * np.linalg.pinv(xtrain)
            yTest = xtest.dot(w)
            predictTest = np.zeros(len(ttest))
            for i in range(0, len(ttest)):
                if yTest[i] < 0:
                    predictTest[i] = 0
                else:
                    predictTest[i] = 1
                    
            plt.plot(ttest[:], 'b.')
            plt.plot(predictTest[:], 'r.')
            plt.show()
            plt.clf()
            
            plt.figure(0)
            plt.figure(1)
            for k in range(0, 9):
                xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
                xtrain = xtrain.astype(float)
                xtest = xtest.astype(float)
                ttrain = ttrain.astype(float)
                ttest = ttest.astype(float)
                
                plt.figure(0)
                plt.subplot(3, 3, k+1)
                plt.plot(xtrain[:,0], xtrain[:,2], 'b.')
                plt.plot(xtest[:,0], xtest[:,2], 'r.')
                
                w = ttrain1 * np.linalg.pinv(xtrain)
                yTest = xtest.dot(w)
                
                predictTest = np.zeros(len(ttest))
                for i in range(0, len(ttest)):
                    if yTest[i] < 0:
                        predictTest[i] = 0
                    else:
                        predictTest[i] = 1
                
                plt.figure(1)
                plt.subplot(3, 3, k+1)
                plt.plot(ttest[:], 'b.')
                plt.plot(predictTest[:], 'r.')
            plt.show()
            plt.clf()     
    ans = input("Συνέχεια(y/n); ")
