import numpy as np
from scipy.special import expit

def lettercode2Strategy(coded, Num_Loc):
	letters = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789;:')
	v = []
	for c in coded:
		v.append(letters.index(c))
	return v

def code2Vector(strategy, Num_Loc):
    size = int(Num_Loc * Num_Loc)
    v = [0] * size
    for i in range(size):
        if i in strategy:
            v[i] = 1
    return v

def sigmoid(x, beta, gamma):
    '''Defines attractiveness and choice functions'''
    return expit(beta * (x - gamma))

def sim_consist(v1, v2):
    '''Returns the similarity based on consistency
	
    v1 and v2 are two 64-bit coded regions'''
    if type(v1) == type(np.nan) or type(v2) == type(np.nan):
        return np.nan
    else:
        assert(len(v1) == 64), 'v1 must be a 64-bit coded region!'
        assert(len(v2) == 64), 'v2 must be a 64-bit coded region!'
        joint = [v1[x] * v2[x] for x in range(len(v1))]
        union = [v1[x] + v2[x] for x in range(len(v1))]
        union = [x/x for x in union if x != 0]
        j = np.sum(np.array(joint))
        u = np.sum(np.array(union))
        if u != 0:
            return float(j)/u
        else:
	        return 1