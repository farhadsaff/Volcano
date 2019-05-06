# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:31:14 2019

@author: fsaff
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:33:28 2019

@author: fsaff
"""

import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
import scipy.io as spio
from scipy.interpolate import interp1d

import numpy as np
from numpy import array
from numpy import polyfit



mat = spio.loadmat('Volcano.mat', squeeze_me=True)

# getting the data

A = mat['A'] # array
A1 = mat['A1'] # structure containing an array
A2 = mat['A2'] # array of structures
A3 = mat['A3']

# getting the axial location from the jet
NN=200
X = np.linspace(0, 23, num=21, endpoint=True)
X1 = np.linspace(0, 23, num=16, endpoint=True)
X2 = np.linspace(0, 23, num=17, endpoint=True)
X3 = np.linspace(0, 23, num=20, endpoint=True)

# Defining the interpolation functions 
F = interp1d(X, A[:,1], kind='cubic')
F1 = interp1d(X1, A1[:,1], kind='cubic')
F2 = interp1d(X2, A2[:,1], kind='cubic')
F3 = interp1d(X3, A3[:,1], kind='cubic')

Fx = interp1d(X, A[:,0], kind='cubic')


xnew = np.linspace(0, 23, num=NN, endpoint=True)

AA=F(xnew).reshape(-1,1)
BB=F1(xnew).reshape(-1,1)
CC=F2(xnew).reshape(-1,1)
DD=F3(xnew).reshape(-1,1)

HH=np.concatenate([DD, CC, BB, AA], axis=1)

xData=np.array([1,1.4,1.96,2.93])
#xData=xData
xData.reshape(-1,1)

#a=0.0
#b=0.0
#c=0.0
#fittedParameters=0.0
#for i in range(100):
#yData[1,:]=HH[1,:]
    
    # function for genetic algorithm to minimize (sum of squared error)
    
#def func(xData, a, b):
    #return a * xData^b
#N=100

a = numpy.empty(NN, dtype=np.float64)
b = numpy.empty(NN, dtype=np.float64)
c = numpy.empty(NN, dtype=np.float64)
AAA=numpy.array(1.0)
BBB=numpy.array(1.0)
CCC=numpy.array([1.0])
#b=numpy.array([1.0,])
#c=numpy.array([1.0,])
#fittedParameters=numpy.array([1,1,1])
#a=a.reshape(5,)
#b=b.reshape(5,)
#c=c.reshape(5,)
#fittedParameters=fittedParameters.reshape(1,)
#a=np.array(a, dtype=np.float64)
#b=np.array(b, dtype=np.float64)
#c=np.array(c, dtype=np.float64) 

#Offset=0
  
def func(x, a, b, Offset): # Sigmoid A With Offset from zunzun.com
        return   ((-a * (x**b))) * Offset

for i in range (0, NN):
 #Offset=0
 yData=HH[i,:]
 

# function for genetic algorithm to minimize (sum of squared error)
 def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = func(xData, *parameterTuple)
    return numpy.sum((yData - val) ** 2.0)


 def generate_Initial_Parameters(): 
     # min and max used for bounds
     maxX = max(xData)
     minX = min(xData)
     maxY = max(yData)
     minY = min(yData)

     parameterBounds = []
     parameterBounds.append([-1, 1]) # seach bounds for a
     parameterBounds.append([-1, 1]) # seach bounds for b
     parameterBounds.append([-1, 1]) # seach bounds for Offset

    # "seed" the numpy random number generator for repeatable results
     result = differential_evolution(sumOfSquaredError, parameterBounds, seed=1)
     return result.x
 
# generate initial parameter values
 geneticParameters = generate_Initial_Parameters()
 #fittedParameters=fittedParameters.reshape(-1,1)
# curve fit the test data
 
 fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters, maxfev=150)
 
 a[(i,)]=np.array([fittedParameters[0,]])
 b[(i,)]=np.array([fittedParameters[1,]])
 c[(i,)]=np.array([fittedParameters[2,]])
 
 T=a*c
 #AAA([i])=a
 #BBB([i])=b
 #CCC([i])=c

 modelPredictions = func(xData, *fittedParameters) 
#AAA[i,1]=a
#BBB[i,1]=b

 absError = modelPredictions - yData
#AAA[i,1]=a
#BBB[i,1]=b

print('Parameters', fittedParameters)

SE = numpy.square(absError) # squared errors
MSE = numpy.mean(SE) # mean squared errors
RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)

print()
    

plt.plot(X, A[:,1], 'o', xnew, AA, '-', xnew, BB, '--',xnew, CC, '--',xnew, DD, '--')
plt.legend(['data', 'cubic'], loc='best')
plt.show()

def ModelAndScatterPlot(graphWidth, graphHeight):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    # first the raw data as a scatter plot
    axes.plot(xData, yData,  'D')

    # create data for the fitted equation plot
    xModel = numpy.linspace(min(xData), max(xData))
    yModel = func(xModel, *fittedParameters)

    # now the model as a line plot
    axes.plot(xModel, yModel)

    axes.set_xlabel('X Data') # X axis data label
    axes.set_ylabel('Y Data') # Y axis data label
    plt.show()
    plt.close('all') # clean up after using pyplot

graphWidth = 800
graphHeight = 600
ModelAndScatterPlot(graphWidth, graphHeight)

#print a[:,:]
#print S['b'][()][:,:] # structures need [()]
#print M[0]['c'][()][:,:]
#print M[1]['c'][()][:,:]

xnew1 = np.linspace(0, 23, num=NN, endpoint=True)

XxX=Fx(xnew1).reshape(-1,1)

#from sklearn.linear_model import LinearRegression 
#lin = LinearRegression() 
  
#lin.fit(XxX[1:39],a[1:39] ) 
b1=b.reshape(-1, 1)
#a1=a.reshape(-1, 1)
#c1=c.reshape(-1, 1)
T1=T.reshape(-1, 1)

#z = np.polyfit(xnew1[1:100], b[1:100], 4)
#coef1z = lin_reg1.z
#zz=z[(4,)]*XxX**4-z[(3,)]*XxX**3+z[(2,)]*XxX**2-z[(1,)]*XxX**1+z[(0,)]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(XxX[1:NN-1])
poly_reg.fit(X_poly, b[1:NN-1])
lin_reg = LinearRegression()
lin_reg.fit(X_poly, b[1:NN-1])

#print (poly_reg)
# prints [[ 1  2  3  4  6  9  8 12 18 27]]
#print (poly_reg.powers_)

#features = DataFrame(X_poly , columns=X_poly.get_feature_names(lin_reg.fit))
#print (features)

coef = lin_reg.coef_




poly_reg2 = PolynomialFeatures(degree = 1)
X_poly2 = poly_reg2.fit_transform(XxX[1:NN-1])
poly_reg2.fit(X_poly2 , T[1:NN-1])
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly2 , T[1:NN-1])

coef2 = lin_reg2.coef_

zzz1=coef[(2,)]*XxX**2+coef[(1,)]*XxX+0.3

zzz2=(coef2[1,]*XxX)-0.06
LLL=zzz2*2.5**zzz1
LLL1=zzz2*1.6**zzz1
LLL2=zzz2*1.8**zzz1
LLL3=zzz2*3**zzz1

NNp=NN-1

#plt.scatter(XxX[1:NNp], a[1:NNp], color='blue')
plt.scatter(XxX[1:NNp], b[1:NNp],color='red')
plt.scatter(XxX[1:NNp], T[1:NNp],color='black')
plt.plot(XxX[1:NNp], lin_reg.predict((X_poly)), color = 'red') 
#plt.plot(XxX[1:NNp], lin_reg1.predict((X_poly1)), color = 'blue') 
plt.plot(XxX[1:NNp], lin_reg2.predict((X_poly2)), color = 'black') 
plt.title('Linear Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 

plt.title('Curve fitting of a,b, and c with regard to x/D')
plt.xlabel('Position level')
plt.ylabel('a,b,c')
plt.show()

plt.plot(XxX, -LLL, color = 'black')
plt.plot(XxX,- LLL1, color = 'red') 
plt.plot(XxX, -LLL2, color = 'blue') 
plt.plot(XxX,- LLL3, color = 'brown') 

plt.title('Curve fitting of entrainment ratio with regard to x/D')
plt.xlabel('Position level')
plt.ylabel('K=1,1.6,1.8,2')
plt.show()
 
plt.show()
def func(x, a, b): # Sigmoid A With Offset from zunzun.com
    return  (-a * (x)^b) 

plt.plot(xnew, AA, color = 'black')
plt.plot(xnew, BB, color = 'red') 
plt.plot(xnew, CC, color = 'blue') 
plt.plot(xnew, DD, color = 'brown') 

plt.title('Interpolation results with regard to x/D')
plt.xlabel('Position level')
plt.ylabel('K=1,1.6,1.8,2')
plt.show()