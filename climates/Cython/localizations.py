# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:16:50 2022

@author: alberto
"""
import numpy as np

def poisson_covariance(theta, alpha):
    """
    

    Parameters
    ----------
    theta : great circle distance matrix
    alpha : cutoff distance.

    Returns
    -------
    weights : localization matrix.

    """
    weights = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if theta[i,j]<alpha:
                weights[i,j] = (max(0,(1 - alpha**2) / (1 - 2*alpha*np.cos(theta[i,j]) + alpha**2)**(3/2)))
            else:
                weights[i,j] = 0
    return weights
    

def spherical(theta,alpha):
    """
    

    Parameters
    ----------
    theta : great circle distance.
    alpha : cutoff distance.

    Returns
    -------
    weights : localization matrix

    """
    weights = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if theta[i,j]<alpha:
                weights[i,j] = (max(0,(1-(theta[i,j]/alpha))**2)*(1+(theta[i,j]/(2*alpha))))
            else:
                weights[i,j] = 0
    return weights

def powered_exp(alpha, theta, nu,tau=0):
    """
    

    Parameters
    ----------
    alpha : > 0
    theta : GD dist
    nu : in (0,1] fractal dimension

    Returns
    -------
    Powered Exponential Covariance.

    """
    weights = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if theta[i,j]<alpha:
                weights[i,j] = (max(0,(1-tau**2)*np.exp(-(theta[i,j]/alpha)**(nu))))
            else:
                weights[i,j] = 0
    return weights
    

def generalized_cauchy(alpha,theta,nu,tau):
    """
    

    Parameters
    ----------
    alpha : > 0.
    theta : GC dist
    nu : in (0,1]
    tau : > 0 

    Returns
    -------
    Generalized Cauchy.

    """
    weights = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if theta[i,j]<alpha:
                weights[i,j] = (max(0,(1+(alpha*theta[i,j])**(nu))**(- tau/nu)))
            else:
                weights[i,j] = 0
    return weights
    

def multiquadric(tau, alpha , theta):
    """
    

    Parameters
    ----------
    tau : in (0,1]
    alpha : > 0 
    theta : GC dist

    Returns
    -------
    Multiquadric covariance.

    """
    weights = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if theta[i,j]<alpha:
                weights[i,j] = (max(0,(1-tau)**(2*alpha)/(1+tau**2 - 2*tau*np.cos(theta[i,j]))**alpha))
            else:
                weights[i,j] = 0
    return weights
    

def sine_power(theta,nu,alpha):
    """
    

    Parameters
    ----------
    theta : GDC.
    nu : in (0,2)
    alpha: range

    Returns
    -------
    Sine Power Covariance.

    """
    weights = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if theta[i,j]<alpha:
                weights[i,j] = max(0,1 - (np.sin(theta[i,j]/2))**nu)
            else:
                weights[i,j] = 0
    return weights
    



def askey(alpha,theta,tau):
    """
    

    Parameters
    ----------
    alpha : > 0 
    theta : GDC
    tau : >= 2.

    Returns
    -------
    Askey.

    """
    
    weights = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if weights[i,j] < alpha:
                weights[i,j] = max(0,(1 - alpha*theta[i,j])**2)
            else:
                weights[i,j]=0
            
    return weights
    


def c2_wendland(tau,alpha,theta):
    """
    

    Parameters
    ----------
    tau : >= 4
    alpha : >= 1/pi
    theta : GCD

    Returns
    -------
    C²-Wendland.

    """
    weights = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if weights[i,j]<alpha:
                weights[i,j] = max(0,(1 + tau*alpha*theta[i,j])*(1-alpha*theta[i,j])**tau)
            else:
                weights[i,j]=0
            
    return weights
    
def c4_wendland(tau,alpha,theta):
    """
    

    Parameters
    ----------
    tau : >= 6
    alpha : >= 1/pi
    theta : GCD

    Returns
    -------
    C⁴-Wendland.

    """
    weights = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if weights[i,j]<alpha:
                weights[i,j] = max(0,(1 + tau*alpha*theta[i,j] + ((tau**2 - 1)/3)*(alpha*theta[i,j])**2)*(1-alpha*theta[i,j])**tau)
            else:
                weights[i,j]=0
            
    return weights

def gaspari_cohn(distances, halfwidth):
    """ Compute Gaspari-Cohn weights from a distance array and
    a given halfwidth """
    r = np.divide(distances, abs(halfwidth))

    # Do the Gaspari Cohn weighting
    weights = np.zeros(r.shape)
    # Less than halfwidth
    weights[r <= 1.0] = ((((-0.25 * r + 0.5) * r + 0.625) * r - 5.0 / 3.0) * r ** 2 + 1.0)[r <= 1.0]
    # Between halfwidth and fullwidth
    weights[(r > 1.0) & (r < 2.0)] = (((((r / 12.0 - 0.5) * r + 0.625) * r + \
                                        5.0 / 3.0) * r - 5.0) * r + 4.0 - \
                                      2.0 / (3.0 * r))[(r > 1.0) & (r < 2.0)]
    return weights

