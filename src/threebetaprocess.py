import numpy as np
from scipy.stats import poisson, beta

def  threebeta(c,b,a,n_max):
    '''Calculate the three-beta process for given parameters.'''
    W=[]
    for i in range(1,n_max):
        C=poisson.rvs(c)
        if C!=0:
            V=np.zeros(shape=(C,i))
            for l in range(i):
                V[:,l]=beta.rvs(1-a,b+(l+1)*a,size=C)
            for j in range(C):
                if i==1:
                    prod=1
                else:
                    prod=np.prod(1-V[j,:i-1])
                w=V[j,i-1]* prod
                W.append(w)
    return np.array(W)
            

def threebeta_vectorized(c, b, a, n_max):
    W = []
    # We can't fully vectorize the outer loop over 'i' directly
    # because C (poisson.rvs) depends on the iteration.
    for i in range(1, n_max):
        C = poisson.rvs(c)
        if C == 0:
            continue

        # Generate all V values for the current 'i' and 'C' at once
        # Create an array of shape parameters for beta distribution
        alpha_params = 1 - a
        beta_params = b + (np.arange(1, i + 1) * a)

        # Draw all beta random variates for the current i and C
        # This will create a (C, i) array of samples
        V = beta.rvs(alpha_params, beta_params, size=(C, i))

        # Calculate the product term: np.prod(1-V[j,:i-1])
        if i == 1:
            prod_terms = np.ones(C) # For i=1, prod is 1 for all C samples
        else:
            # Calculate products across the first i-1 columns
            # This is equivalent to np.cumprod(1 - V[:, :i-1], axis=1)[:, -1]
            prod_terms = np.prod(1 - V[:, :i-1], axis=1)
        
        # Calculate w = V[j,i-1] * prod
        w_values = V[:, i-1] * prod_terms
        W.extend(w_values) # Use extend for efficiency

    return np.array(W)