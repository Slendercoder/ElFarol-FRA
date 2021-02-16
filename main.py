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
Pl = 5 # number of players
rounds = 60 # number of rounds
groups = 50 # number of groups
gameParameters = [p, Pl, rounds, groups]
measures = '13'
TO_FILE = True
non_shaky_hand = 0.88

playerParameters1 =  {'ALL': 0.1} # Bias towards ALL
playerParameters1['NOTHING'] = 0.1 # Bias towards NOTHING
playerParameters1['3-GO-2-STAY'] = 0.1 # Bias towards 3-GO-2-STAY
playerParameters1['alpha'] = 100 # How much the focal region augments attractiveness
playerParameters1['beta'] = 30 # Amplitude of the WSLS sigmoid function
playerParameters1['gamma'] = 31 # Position of the WSLS sigmoid function
playerParameters1['delta'] = 100 # How much the added FRA similarities augments attractiveness
playerParameters1['epsilon'] = 30 # Amplitude of the FRA sigmoid function
playerParameters1['zeta'] = 0.7 # Position of the FRA sigmoid function

playerParameters2 =  {'ALL': 0.1} # Bias towards ALL
playerParameters2['NOTHING'] = 0.1 # Bias towards NOTHING
playerParameters2['3-GO-2-STAY'] = 0.1 # Bias towards 3-GO-2-STAY
playerParameters2['alpha'] = 100 # How much the focal region augments attractiveness
playerParameters2['beta'] = 30 # Amplitude of the WSLS sigmoid function
playerParameters2['gamma'] = 31 # Position of the WSLS sigmoid function
playerParameters2['delta'] = 100 # How much the added FRA similarities augments attractiveness
playerParameters2['epsilon'] = 30 # Amplitude of the FRA sigmoid function
playerParameters2['zeta'] = 0.7 # Position of the FRA sigmoid function

print("****************************")
print('Starting simulation')
print("****************************")
print('--- Model parameters ----')
print('--- Player 1 ----')
print('Bias towards ALL: ', playerParameters1['ALL'])
print('Bias towards NOTHING: ', playerParameters1['NOTHING'])
print('Bias towards 3-GO-2-STAY: ', playerParameters1['3-GO-2-STAY'])
print('alpha: ', playerParameters1['alpha'])
print('beta: ', playerParameters1['beta'])
print('gamma: ', playerParameters1['gamma'])
print('delta: ', playerParameters1['delta'])
print('epsilon: ', playerParameters1['epsilon'])
print('zeta: ', playerParameters1['zeta'])
print("\n")
print('--- Player 2 ----')
print('Bias towards ALL: ', playerParameters2['ALL'])
print('Bias towards NOTHING: ', playerParameters2['NOTHING'])
print('Bias towards 3-GO-2-STAY: ', playerParameters2['3-GO-2-STAY'])
print('alpha: ', playerParameters2['alpha'])
print('beta: ', playerParameters2['beta'])
print('gamma: ', playerParameters2['gamma'])
print('delta: ', playerParameters2['delta'])
print('epsilon: ', playerParameters2['epsilon'])
print('zeta: ', playerParameters2['zeta'])
print("****************************")
print('--- Game parameters ---')
print('Threshold: ', gameParameters[0])
print('Number of players: ', gameParameters[1])
print('Number of rounds: ', gameParameters[2])
print('Number of groups: ', gameParameters[3])
print("\n")

E = DL.Experiment(gameParameters, [playerParameters1, playerParameters2], non_shaky_hand=non_shaky_hand)
if TO_FILE:
        with open('temp.csv', 'w') as dfile:
            head = 'index,Group,Round,Player,Decision,Score,Strategy\n'
            dfile.write(head)
            dfile.close()
        E.df = pd.read_csv('temp.csv')
E.run_simulation()
if TO_FILE:
    E.df = pd.read_csv('temp.csv')
#M = Measures.Measuring(data=E.df, Num_Loc=rounds, TOLERANCE=0)
#E.df = M.get_measures(measures)
if TO_FILE: os.remove("temp.csv")
E.save()