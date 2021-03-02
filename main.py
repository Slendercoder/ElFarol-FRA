print('Importing packages...')
import EmergenceDCL as DL
import Measures
import pandas as pd
import os
print('Done!')

##########################################################################
#
#  Simulation starts here
#
##########################################################################

# Create experiment
p = 0.6 # threshold
Pl = 2 # number of players
rounds = 50 # number of rounds
groups = 1 # number of groups
gameParameters = [p, Pl, rounds, groups]
measures = '13'
TO_FILE = True

# 5 players:
# playerParameters = {'ALL': 0.05} # Bias towards ALL
# playerParameters['NOTHING'] = 0.05 # Bias towards NOTHING
# playerParameters['ALTER1'] = 0.05 # Bias towards 3-GO-2-STAY
# playerParameters['ALTER2'] = 0.05 # Bias towards 3-GO-2-STAY
# playerParameters['ALTER3'] = 0.05 # Bias towards 3-GO-2-STAY
# playerParameters['ALTER4'] = 0.05 # Bias towards 3-GO-2-STAY
# playerParameters['ALTER5'] = 0.05 # Bias towards 3-GO-2-STAY

# 2 players:
playerParameters = {}
playerParameters['ALTER1'] = 0.5 # Bias towards 3-GO-2-STAY
playerParameters['ALTER2'] = 0.5 # Bias towards 3-GO-2-STAY

playerParameters['alpha'] = 100 # How much the focal region augments attractiveness
playerParameters['beta'] = 500 # Amplitude of the WSLS sigmoid function
playerParameters['gamma'] = -0.5 # Position of the WSLS sigmoid function
playerParameters['delta'] = 0 # How much the added FRA similarities augments attractiveness
playerParameters['epsilon'] = 0 # Amplitude of the FRA sigmoid function
playerParameters['zeta'] = 0 # Position of the FRA sigmoid function

print("****************************")
print('Starting simulation')
print("****************************")
print('--- Model parameters ----')
print('alpha: ', playerParameters['alpha'])
print('beta: ', playerParameters['beta'])
print('gamma: ', playerParameters['gamma'])
print('delta: ', playerParameters['delta'])
print('epsilon: ', playerParameters['epsilon'])
print('zeta: ', playerParameters['zeta'])
print("\n")
print("****************************")
print('--- Game parameters ---')
print('Threshold: ', gameParameters[0])
print('Number of players: ', gameParameters[1])
print('Number of rounds: ', gameParameters[2])
print('Number of groups: ', gameParameters[3])
print("\n")

E = DL.Experiment(gameParameters, playerParameters)
if TO_FILE:
    with open('temp.csv', 'w') as dfile:
        head = 'index,Group,Round,Player,Decision,Score,Strategy\n'
        dfile.write(head)
    E.df = pd.read_csv('temp.csv')
E.run_simulation()
if TO_FILE:
    E.df = pd.read_csv('temp.csv')
    os.remove("temp.csv")
#M = Measures.Measuring(data=E.df, Num_Loc=rounds, TOLERANCE=0)
#E.df = M.get_measures(measures)
E.save()