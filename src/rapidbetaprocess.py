import numpy as np
from tqdm import tqdm
from src.threebetaprocess import threebeta
from scipy.special import gamma


def rapidbeta(eta,ksi,n_max,k_max):
    '''Calculate the rapid-beta process for given parameters.'''
    W = []
    S = np.logspace(-15, 0, k_max) 
    S = 1 - S[::-1]
    for k in tqdm(range(0, k_max-1)):
        c=eta*gamma(1-S[k+1])*gamma(ksi)/gamma(1-S[k])/gamma(1-S[k+1]+ksi)
        b=ksi-S[k+1]
        a=S[k+1]
        proposalW= threebeta(c, b, a, n_max)
        C= len(proposalW)
        if C != 0:
            proposalS=np.random.uniform(low=S[k], high=S[k+1], size=C)
            Acceptance=gamma(1-S[k])/gamma(1-proposalS)*proposalW**(S[k+1]-proposalS)
            U=np.random.uniform(size=C)
            Accepted=U<Acceptance
            print(np.sum(Accepted),"/",C)
            W.extend(proposalW[Accepted])
    return np.array(W)

    