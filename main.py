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
rounds = 10 # number of rounds
groups = 1 # number of groups
gameParameters = [p, Pl, rounds, groups]
measures = '13'
TO_FILE = True

playerParameters1 =  {'ALL': 0.05} # Bias towards ALL
playerParameters1['NOTHING'] = 0.05 # Bias towards NOTHING
playerParameters1['ALTER1'] = 0.05 # Bias towards 3-GO-2-STAY
playerParameters1['ALTER2'] = 0.05 # Bias towards 3-GO-2-STAY
playerParameters1['ALTER3'] = 0.05 # Bias towards 3-GO-2-STAY
playerParameters1['ALTER4'] = 0.05 # Bias towards 3-GO-2-STAY
playerParameters1['ALTER5'] = 0.05 # Bias towards 3-GO-2-STAY
playerParameters1['alpha'] = 100 # How much the focal region augments attractiveness
playerParameters1['beta'] = 30 # Amplitude of the WSLS sigmoid function
playerParameters1['gamma'] = 31 # Position of the WSLS sigmoid function
playerParameters1['delta'] = 100 # How much the added FRA similarities augments attractiveness
playerParameters1['epsilon'] = 30 # Amplitude of the FRA sigmoid function
playerParameters1['zeta'] = 0.7 # Position of the FRA sigmoid function

playerParameters2 = playerParameters1

print("****************************")
print('Starting simulation')
print("****************************")
print('--- Model parameters ----')
print('--- Player 1 ----')
print('Bias towards ALL: ', playerParameters1['ALL'])
print('Bias towards NOTHING: ', playerParameters1['NOTHING'])
print('Bias towards ALTER1: ', playerParameters1['ALTER1'])
print('Bias towards ALTER2: ', playerParameters1['ALTER2'])
print('Bias towards ALTER3: ', playerParameters1['ALTER3'])
print('Bias towards ALTER4: ', playerParameters1['ALTER4'])
print('Bias towards ALTER5: ', playerParameters1['ALTER5'])
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
print('Bias towards ALTER1: ', playerParameters2['ALTER1'])
print('Bias towards ALTER2: ', playerParameters2['ALTER2'])
print('Bias towards ALTER3: ', playerParameters2['ALTER3'])
print('Bias towards ALTER4: ', playerParameters2['ALTER4'])
print('Bias towards ALTER5: ', playerParameters2['ALTER5'])
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

E = DL.Experiment(gameParameters, [playerParameters1, playerParameters2])
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