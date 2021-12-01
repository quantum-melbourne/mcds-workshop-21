# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:14:00 2021

@author: jamie
"""

# Utility Functions that we don't need to show in the Jupyter notebook
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from scipy import stats
from qiskit.quantum_info import Statevector, Operator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
import cmath

def plot_bloch(theta, phi, labels):

    # Input: theta, phi - Arrays containing the theta and phi coordinates for the bloch sphere representation of a statevector for each datapoint.
    # Output : None - Plots a 3D graph
    r = 1.0
    
    xs = r*np.sin(theta)*np.cos(phi)
    ys = r*np.sin(theta)*np.sin(phi)
    zs = r*np.cos(theta)
    
    " Create the sphere outline "
    
    phi_outline = np.linspace(0, np.pi, 100)
    theta_outline = np.linspace(0, 2*np.pi, 100)
    phi_outline, theta_outline = np.meshgrid(phi_outline, theta_outline)
    
    x = np.sin(phi_outline) * np.cos(theta_outline)
    y = np.sin(phi_outline) * np.sin(theta_outline)
    z = np.cos(phi_outline)
    
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color="w", rstride=1, cstride=1, linewidth=0, alpha=0.3)
    
    
    " Plot our pointson the graph"
    
    color= ['red' if l == 0 else 'green' for l in labels]
    ax.scatter(xs,ys,zs,color=color,s=20)
    
    
    
    plt.show()
    
    
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               #levels=[-1, 0, 1], alpha=0.5,
               levels=[ 0 ], alpha=0.5,
               #linestyles=['--', '-', '--'])
               linestyles=['-'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
# Find accuracy of model
def accuracy_score(model, x, labels):
    predictions = model.predict(x)
    predictions = predictions.reshape(len(predictions), 1)
    labels = labels.reshape(len(labels), 1)
    correct = predictions == labels
    accuracy = sum(correct) / len(correct)
    print("Accuracy score is ", np.round(accuracy[0],3)*100, "%")