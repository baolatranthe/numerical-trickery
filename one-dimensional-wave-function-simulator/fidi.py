import numpy as np
from math import factorial as factorial

def diff_coeff(l, m, n):
    
    """
    
    This function generates the 'n-th' coefficient of a 1-D finite difference
    scheme for an arbitrary order 'l' with an offset 'm'. The 'offset' is 
    defined as the point within the stencil where the approximation for the
    derivative is calculated. For instance, a finite difference scheme of an 
    l-point stencil with m=(l-1)/2 and l is odd, is called a 'central' finite 
    difference. For the 'forward' and 'backward' difference schemes, m is 0 
    and l-1, respectively.
    
    """
    
    def factorial(k):
        if type(k) is not int:
            print("Invalid input! The function 'factorial' only accepts non-negative integers.")
        elif k < 0:
            return None
        elif k == 0:
            return 1
        else:
            return k*factorial(k-1)
    
    coeff = 0
    if (m >= l) or (n >= l):
        print("Invalid inputs! The function 'diff_coeff' requires 0 < m < l and 0 < n < l.")
        return None
    elif m != n:
        coeff = (-1)**(m+n)/(m-n) * factorial(m)*factorial(l-m-1)/factorial(n)/factorial(l-n-1)
    else:
        for i in range(l):
            if i == m:
                continue
            else:
                coeff += -diff_coeff(l, m, i)
    
    return coeff

def bounded_diff(M, m=1):
    
    """
    
    This function generates the difference matrix for a bounded
    domain of M grid points. with a difference scheme of order 2m+1.
    
    """
    
    matrix = np.zeros((M, M))
    l = 2*m+1
    
    if (M < l) or (type(M) is not int) or (type(m) is not int):
        print("Invalid inputs! Expecting integers for M and m, where M >= 2m+1.")
        return None
    else:
        for i in range(M):
            for j in range(M):
                if (i < m) and (j < l):
                    matrix[i, j] = diff_coeff(l, i, j)
                elif (M-i <= m) and (M-j <= l):
                    matrix[i, j] = diff_coeff(l, i+l-M, j+l-M)
                elif np.abs(j-i) <= m:
                    matrix[i, j] = diff_coeff(l, m, j-i+m)
                else:
                    continue
    
    return matrix

def periodic_diff(M, m=1):
    
    """
    
    This function generates the difference matrix for a periodic
    domain of M grid points. with a difference scheme of order 2m+1.
    
    """
    
    matrix = np.zeros((M, M))
    l = 2*m+1
    
    if (M < l) or (type(M) is not int) or (type(m) is not int):
        print("Invalid inputs! Expecting integers for M and m, where M >= 2m+1.")
        return None
    else:
        for i in range(M):
            for j in range(M):
                if np.abs(j-i) <= m:
                    matrix[i, j] = diff_coeff(l, m, j-i+m)
                elif M-np.abs(j-i) <= m:
                    matrix[i, j] = diff_coeff(l, m, int(np.sign(j-i)*(np.abs(j-i)-M)+m))
                else:
                    continue
    
    return matrix

def spacing(i, f, n, bound=True, periodic=False):
    
    """
    
    Create linear n spacing points from i to f.
    Bound spacing includes the initial and final points.
    Periodic spacing includes initial point but exclude final point.
    Neither spacing excludes both i and f.
    
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

