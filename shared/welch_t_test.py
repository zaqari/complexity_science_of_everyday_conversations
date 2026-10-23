import numpy as np
from scipy.stats import t

def welchT(
        mu1: float,
        n1: int|float,
        sd1: float,
        mu2: float,
        n2: int|float,
        sd2: float
):

    vnum = (((sd1**2) / n1) + ((sd2**2) / n2))**2
    vdenom = ((sd1**4)/((n1**2)*(n1-1))) + ((sd2**4)/((n2**2)*(n2-1)))
    dof = vnum/vdenom

    tstat = (mu1 - mu2) / np.sqrt(((sd1/np.sqrt(n1))**2) + ((sd2/np.sqrt(n2))**2))

    return tstat, dof, 1-t.cdf(tstat.__abs__(),df=dof)