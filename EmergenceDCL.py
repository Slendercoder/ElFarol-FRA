from random import choices, uniform, randint
from math import floor
import numpy as np
import pandas as pd
import os
import FRA

DEB = False

#################################
# FUNCTIONS
################################

# Define players
class player :
	'''Object defining a player.'''

	def __init__(self, Ready, Decision, Choice, Where, Score, Accuracy, Name, modelParameters):
		self.ready = Ready
		self.decision = Decision
		self.choice = Choice
		self.where = Where
		self.score = Score
		self.accuracy = Accuracy
		self.name = Name
		self.parameters = modelParameters
		self.regionsNames = ['RS', \
		           'ALL', \
		           'NOTHING', \
		           'ALTERNATE']
		self.regionsCoded = ['abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789;:', # ALL
		                  '', # NOTHING
						  'abcfghklmpqruvwzABEFGJKLOPQTUVYZ034589;', # ALTER1
						  'abefgjklopqtuvyzADEFIJKNOPSTUXYZ234789', # ALTER2
						  'adefijknopstuxyzCDEHIJMNORSTWXY123678:', # ALTER3
						  'cdehijmnorstwxyBCDGHILMNQRSVWX012567;:', # ALTER4
						  'bcdghilmnqrsvwxABCFGHKLMPQRUVWZ014569;:' # ALTER5
		                  ]
		self.strategies = []
		self.regions = []
		self.complements = []

	def make_decision(self, Num_Loc):
		attractiveness = self.attract(Num_Loc)
		sum = np.sum(attractiveness)
		probabilities = [x/sum for x in attractiveness]
		newChoice = choices(range(len(self.parameters)-5), weights=probabilities)[0]
		self.choice = newChoice

	def attract(self, Num_Loc, DEB=False):
		wALL = float(self.parameters['ALL'])
		wNOTHING = float(self.parameters['NOTHING'])
		wALTERNATE = float(self.parameters['3-GO-2-STAY'])
		wRS = 1 - np.sum(np.array([wALL, wNOTHING, wALTERNATE]))
		assert(wRS > 0), "Incorrect biases! Sum greater than 1"

		alpha = float(self.parameters['alpha']) # for how much the focal region augments attractiveness
		beta = float(self.parameters['beta']) # amplitude of the WSLS sigmoid function
		gamma = float(self.parameters['gamma']) # position of the WSLS sigmoid function

		delta = float(self.parameters['delta']) # for how much the added FRA similarities augments attractiveness
		epsilon = float(self.parameters['epsilon']) # amplitude of the FRA sigmoid function
		zeta = float(self.parameters['zeta']) # position of the FRA sigmoid function

		# start from biases
		attractiveness = [wRS, wALL, wNOTHING, wALTERNATE]
		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('Player', self.name)
			print('attractiveness before WS and FRA\n', attactPrint)

		# Adding 'Win Stay'
		if self.choice != 0:
			attractiveness[self.choice] += alpha * FRA.sigmoid(self.score, beta, gamma)

		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('attractiveness with WS\n', attactPrint)

		# Adding 'FRA'
		visited = FRA.code2Vector(self.where, Num_Loc)
		sims1 = [0] + [FRA.sim_consist(visited, x) for x in self.regions]
		overlap = FRA.code2Vector(self.score, Num_Loc) # Replace for joint
		sims2 = [0] + [FRA.sim_consist(overlap, x) for x in self.complements]
		sims2[0] = 0 # ALL's complement, NOTHING, does not repel to ALL
		FRAsims = np.add(sims1, sims2)
		attractiveness = np.add(attractiveness, [delta * FRA.sigmoid(x, epsilon, zeta) for x in FRAsims])

		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('attractiveness with FRA\n', attactPrint)

		return attractiveness

# Define Experiment Object
class Experiment :
	'''Object defining the experiment and simulation'''

	def __init__(self, gameParameters, modelParameters, non_shaky_hand=1):
		assert(len(gameParameters) == 4), "Game parameters incorrect length!"
		self.gameParameters = gameParameters
		self.modelParameters = modelParameters
		self.non_shaky_hand = non_shaky_hand
		# Create data frame
		cols = ['Group', 'Round', 'Player','Decision']
		cols += list(range(self.gameParameters[2]))
		cols += ['Score', 'Strategy']
		self.df = pd.DataFrame(columns=cols)

	def run_group(self, TO_FILE=True):

		p = self.gameParameters[0] # threshold (usually 0.6)
		Pl = self.gameParameters[1] # number of players (usually 5)
		rounds = self.gameParameters[2] # number of rounds (usually 60)

		# Create players
		Players = []
		for k in range(0, Pl):
			Players.append(player(False, "", 0, [], [], 0, False, int(uniform(0, 1000000)), self.modelParameters[k]))

		# Start the rounds
		for i in range(1, rounds+1):
			# Playing round i
			Num_Loc = i

			#Initializing players for round
			for pl in Players:
				pl.decision = ""
				pl.ready = False
				pl.score = 0
				pl.accuracy = False
				pl.strategies = [FRA.lettercode2Strategy(x, i) for x in pl.regionsCoded]
				pl.regions = [FRA.code2Vector(x, i) for x in pl.strategies]
				pl.complements = [[1 - x for x in sublist] for sublist in pl.regions]
			
			# Determine players' chosen region
			chosen_strategies = []
			for k in range(0, Pl):
				chosen = Players[k].choice
				if chosen == 0:
					n = randint(1, Num_Loc)
					chosen_strategies.append(list(np.random.choice(Num_Loc, n, replace=False)))
				else:
					chosen_strategies.append(Players[k].strategies[chosen - 1])
				chosen_strategies[k] = self.shake(chosen_strategies[k])

			# Player decides whether to assist or not
			for k in range(0, Pl):
				if Num_Loc-1 in chosen_strategies[k]:
					search_place = Num_Loc-1
					Players[k].where.append(search_place)
					Players[k].decision = 1
				else:
					Players[k].decision = 0

			# Get results and store data in dataframe (returns players with updated scores)
			Players = self.round2dataframe(Players, i, TO_FILE)

			# Players determine their next strategies
			for k in range(0,Pl):
				Players[k].make_decision(Num_Loc+1)
	
	def shake(self, strategy):
		if uniform(0, 1) > self.non_shaky_hand:
			p = 2
			outs = np.random.choice(strategy, p) if len(strategy) > 0 else []
			complement = [i for i in range(64) if i not in strategy]
			ins = np.random.choice(complement, p) if len(complement) > 0 else []
			return [i for i in strategy if i not in outs] + list(ins)
		else:
			return strategy

	def run_simulation(self):
		iters = self.gameParameters[3] # number of experiments in a set
		for g in range(0, iters):
			print("****************************")
			print("Running group no. ", g + 1)
			print("****************************\n")
			self.run_group()
	
	def round2dataframe(self, Players, round, TO_FILE):
		Num_Loc = self.gameParameters[2]
		# Create row of data as dictionary
		row_of_data = {}
		# Create group name
		group = ''
		for pl in Players: group += str(pl.name)[:5]
		# Determine whether bar was overcrowded
		overcrowded = len([p for p in Players if p.decision == 1])/len(Players) > self.gameParameters[0]
		# Save data per player
		for k in range(0, len(Players)):
			# Determine individual scores
			if overcrowded:
				# Bar was overcrowded
				if Players[k].decision == 0:
					# Player k's decision is Correct
					Players[k].accuracy = True
				else:
					# Player k's decision is Incorrect
					Players[k].accuracy = False
					Players[k].score -= 1
			else:
				# Bar was not overcrowded
				if Players[k].decision == 0:
					# Player k's decision is Incorrect
					Players[k].accuracy = False
				else:
					# Player k's decision is Correct
					Players[k].accuracy = True
					Players[k].score += 1
			row_of_data['Group'] = [group]
			row_of_data['Round'] = [round]
			row_of_data['Player'] = [Players[k].name]
			row_of_data['Decision'] = [Players[k].decision]
			row_of_data['Score'] = [Players[k].score]
			row_of_data['Strategy'] = [Players[k].choice]
			# Add data to dataFrame
			dfAux = pd.DataFrame.from_dict(row_of_data)
			# Keeping the order of columns
			dfAux = dfAux[['Group','Round','Player','Decision','Score','Strategy']]

			if TO_FILE:
				with open('temp.csv', 'a') as f:
					dfAux.to_csv(f, header=False)
			else:
				self.df = self.df.append(dfAux, ignore_index = True)

		return Players

	def save(self):
		count = 0
		file_name = './Data/output' + str(count) + '.csv'
		while os.path.isfile(file_name):
			count += 1
			file_name = './Data/output' + str(count) + '.csv'
		self.df.to_csv(file_name, index=False)
		print('Data saved to ' + file_name)