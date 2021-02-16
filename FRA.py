import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
from random import randint
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

def distance(k, i):
    '''Returns similarity between regions k and i

    Input: k, which is a region coded as a vector of 0s and 1s of length 64
           i, which is a region coded as a vector of 0s and 1s of length 64
           o, which is a parameter for the exponential
    Output: number representing the similarity between k and i'''
    return np.abs(np.subtract(k, i)).sum()

def draw_round(player1, player2, titulo, Num_Loc):

    # Initializing Plot
    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=2, nrows=2)#, height_ratios=[3, 1, 1, 1])
    fig.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0.1, hspace=0.2)

    ax0 = fig.add_subplot(spec[0,0])
    ax1 = fig.add_subplot(spec[0,1])
    ax2 = fig.add_subplot(spec[1,0])
    ax3 = fig.add_subplot(spec[1,1])

    ax0.set_title('Player 1')
    ax0.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)
    ax1.set_title('Player 2')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.set_yticklabels([])
    ax2.set_ylabel('Probabilities', fontsize=8)
    ax3.yaxis.tick_right()

    # Ploting regions
    reg1 = code2Vector(player1.where, Num_Loc)
    reg2 = code2Vector(player2.where, Num_Loc)
    step = 1. / Num_Loc
    tangulos1 = []
    tangulos2 = []
    for j in range(0, Num_Loc * Num_Loc):
        x = int(j) % Num_Loc
        y = (int(j) - x) / Num_Loc
        by_x = x * step
        by_y = 1 - (y + 1) * step
        if reg1[j] == 1:
            tangulos1.append(patches.Rectangle(*[(by_x, by_y), step, step],\
			facecolor="black", alpha=1))
        if reg2[j] == 1:
            tangulos2.append(patches.Rectangle(*[(by_x, by_y), step, step],\
			facecolor="black", alpha=1))
        if reg1[j] == 1 and reg2[j] == 1:
            tangulos1.append(patches.Rectangle(*[(by_x, by_y), step, step],\
			facecolor="red", alpha=1))
            tangulos2.append(patches.Rectangle(*[(by_x, by_y), step, step],\
			facecolor="red", alpha=1))

    for t in tangulos1:
        ax0.add_patch(t)

    for t in tangulos2:
        ax1.add_patch(t)

    # Find probabilities
    attractiveness = player1.attract()
    sum = np.sum(attractiveness)
    frasPL1 = [x/sum for x in attractiveness]
    attractiveness = player2.attract()
    sum = np.sum(attractiveness)
    frasPL2 = [x/sum for x in attractiveness]

    # Plot probabilities
    regions_names = ['RS','A','N','B','T','L','R','I','O']
    ax2.set_ylim(0,max(1,max(frasPL1)))
    ax3.set_ylim(0,max(1,max(frasPL1)))
    ax2.bar(regions_names, frasPL1)
    ax3.bar(regions_names, frasPL2)
    threshold = frasPL1[0]
    ax2.axhline(y=threshold, linewidth=1, color='k')
    threshold = frasPL2[0]
    ax3.axhline(y=threshold, linewidth=1, color='k')

    fig.suptitle(titulo)
    plt.show()

def minDist2Focal(r, regionsCoded):
	'''Returns closest distance to focal region

	Input: r, which is a region coded as a vector of 0s and 1s of length 64
	Output: number representing the closest distance'''
	distances = [distance(r, k) for k in regionsCoded]
	return min(distances)

def new_random_strategy(Num_Loc):
    '''Creates a new random strategy to explore grid'''
    n = randint(2,Num_Loc * Num_Loc - 2)
    return list(np.random.choice(Num_Loc * Num_Loc, n))

def numberRegion(r):
	if r == 'RS':
		return 0
	elif r == 'ALL':
		return 1
	elif r == 'NOTHING':
		return 2
	elif r == 'BOTTOM':
		return 3
	elif r == 'TOP':
		return 4
	elif r == 'LEFT':
		return 5
	elif r == 'RIGHT':
		return 6
	elif r == 'IN':
		return 7
	elif r == 'OUT':
		return 8

def nameRegion(r):
	if r == 0 or r == 9:
		return 'RS'
	elif r == 1:
		return 'ALL'
	elif r == 2:
		return 'NOTHING'
	elif r == 3:
		return 'BOTTOM'
	elif r == 4:
		return 'TOP'
	elif r == 5:
		return 'LEFT'
	elif r == 6:
		return 'RIGHT'
	elif r == 7:
		return 'IN'
	elif r == 8:
		return 'OUT'

def classify_region(r, TOLERANCE, regions):
	'''Returns name of closest region

	Input: r, which is a region coded as a vector of 0s and 1s of length 64'''
	distances = [distance(list(r), k) for k in regions]
	value = np.min(distances)
	indexMin = np.argmin(distances)
	if value <= TOLERANCE:
		return(nameRegion(indexMin + 1))
	else:
		return('RS')

def maxSim2Focal(r, Num_Loc, regions):
    '''Returns maximum similarity (BASED ON CONSISTNECY) to focal region

    Input: r, which is a region coded as a vector of 0s and 1s of length 64
    Output: number representing the highest similarity'''

    similarities = [sim_consist(a,r) for a in regions]
    value = np.max(np.array(similarities))
    return(value)