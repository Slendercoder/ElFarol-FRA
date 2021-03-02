from scipy.special import expit

def sigmoid(x, beta, gamma):
    '''Defines attractiveness and choice functions'''
    return expit(beta * (x - gamma))

def distance(v1, v2):
    '''Returns the distance between two regions'''
    result = 0
    for i in range(len(v1)):
        result += abs(v1[i]-v2[i])
    return result