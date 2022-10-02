import numpy as np
from math import factorial
from scipy import sparse
from tqdm.notebook import tqdm

# Finite Difference

def diff_coeff(l, m, n, d='First'):
    
    """
    
    This function generates the 'n-th' coefficient of a 1-D finite difference
    scheme for an arbitrary order 'l' with an offset 'm'. The 'offset' is 
    defined as the point within the stencil where the approximation for the
    derivative is calculated. For instance, a finite difference scheme of an 
    l-point stencil with m=(l-1)/2 and l is odd, is called a 'central' finite 
    difference. For the 'forward' and 'backward' difference schemes, m is 0 
    and l-1, respectively.
    Return: scalar
    
    """
    
    coeff = 0
    if (m >= l) or (n >= l):
        print("Invalid inputs! The function 'diff_coeff' requires 0 < m < l and 0 < n < l.")
        return None
    elif d == 'First':
        if m != n:
            coeff = (-1)**(m+n)/(m-n) * factorial(m)*factorial(l-m-1)/factorial(n)/factorial(l-n-1)
        else:
            for i in range(l):
                if i == m:
                    continue
                else:
                    coeff += -diff_coeff(l, m, i)
    elif d == 'Second':
        for i in range(l):
            coeff += diff_coeff(l, m, i)*diff_coeff(l, i, n)
    else:
        print("Input for 'd' is not recognized. Valid inputs: d='First' or d='Second'")
        return None
    
    return coeff

def diff(M, m=1, d='First', bc='Bounded'):
    
    """
    
    This function generates the 'd-th' difference matrix with a domain of M grid points
    and a difference scheme of order 2m+1, where M >= 2m+1. The boundary condition can
    be set to either bc='Bounded' or bc='Periodic'.
    
    d='First': first finite difference (1st derivative approximation)
    d='Sencond': second finite difference (2nd derivative approximation)
    
    Return: scipy sparse COO matrix
    
    """
    
    matrix = np.zeros((M, M))
    l = 2*m+1
    I = []
    J = []
    V = []
    
    if (M < l) or (type(M) is not int) or (type(m) is not int):
        print("Invalid inputs! Expecting integers for M and m, where M >= 2m+1.")
        return None
    elif bc == 'Bounded':
        print('Generating difference matrix...')
        for i in tqdm(range(M)):
            for j in range(M):
                if np.abs(j-i) <= m:
                    I.append(i)
                    J.append(j)
                    V.append(diff_coeff(l, m, j-i+m, d))
                else:
                    continue
    elif bc == 'Periodic':
        print('Generating difference matrix...')
        for i in tqdm(range(M)):
            for j in range(M):
                if np.abs(j-i) <= m:
                    I.append(i)
                    J.append(j)
                    V.append(diff_coeff(l, m, j-i+m, d))
                elif M-np.abs(j-i) <= m:
                    I.append(i)
                    J.append(j)
                    V.append(diff_coeff(l, m, int(np.sign(j-i)*(np.abs(j-i) - M) + m), d))
                else:
                    continue
    else:
        print("Invalid boundary condition. Use the following: bc='Bounded' or bc='Periodic'")
        return None
    
    return sparse.coo_array((V,(I,J)), shape=(M, M))

def spacing(i, f, n, bound=True, periodic=False):
    
    """
    
    Create linear n spacing points from i to f.
    Bound spacing includes the initial and final points.
    Periodic spacing includes initial point but exclude final point.
    Neither spacing excludes both i and f.
    Return: numpy array
    
    """
    
    if bound and not periodic:
        step_size = (f-i)/(n-1)
        init = i
    elif periodic and not bound:
        step_size = (f-i)/n
        init = i
    else:
        step_size = (f-i)/(n+1)
        init = i + step_size
        
    array = np.zeros(n)
    for step in range(n):
        array[step] = init + step*step_size
    
    return array