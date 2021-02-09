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
		           'BOTTOM', \
		           'TOP', \
		           'LEFT', \
		           'RIGHT', \
		           'IN', \
		           'OUT']
		self.regionsCoded = ['abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789;:', # ALL
		                  '', # NOTHING
		                  'GHIJKLMNOPQRSTUVWXYZ0123456789;:', # BOTTOM
		                  'abcdefghijklmnopqrstuvwxyzABCDEF', # TOP
		                  'abcdijklqrstyzABGHIJOPQRWXYZ4567', # LEFT
		                  'efghmnopuvwxCDEFKLMNSTUV012389;:', # RIGHT
		                  'jklmnorstuvwzABCDEHIJKLMPQRSTUXYZ012', # IN
		                  'abcdefghipqxyFGNOVW3456789;:' # OUT
		                  ]
		self.strategies = [FRA.lettercode2Strategy(x, 8) for x in self.regionsCoded]
		self.regions = [FRA.code2Vector(x, 8) for x in self.strategies]
		self.complements = [[1 - x for x in sublist] for sublist in self.regions]

	def make_decision(self):
		attractiveness = self.attract()
		sum = np.sum(attractiveness)
		probabilities = [x/sum for x in attractiveness]
		newChoice = choices(range(9), weights=probabilities)[0]
		self.choice = newChoice

	def attract(self, DEB=False):
		wALL = float(self.parameters['ALL'])
		wNOTHING = float(self.parameters['NOTHING'])
		wBOTTOM = float(self.parameters['T-B-L-R'])
		wTOP = float(self.parameters['T-B-L-R'])
		wLEFT = float(self.parameters['T-B-L-R'])
		wRIGHT = float(self.parameters['T-B-L-R'])
		wIN = float(self.parameters['IN-OUT'])
		wOUT = float(self.parameters['IN-OUT'])
		wRS = 1 - np.sum(np.array([wALL, wNOTHING, wBOTTOM, wTOP, wLEFT, wRIGHT, wIN, wOUT]))
		assert(wRS > 0), "Incorrect biases! Sum greater than 1"

		alpha = float(self.parameters['alpha']) # for how much the focal region augments attractiveness
		beta = float(self.parameters['beta']) # amplitude of the WSLS sigmoid function
		gamma = float(self.parameters['gamma']) # position of the WSLS sigmoid function

		delta = float(self.parameters['delta']) # for how much the added FRA similarities augments attractiveness
		epsilon = float(self.parameters['epsilon']) # amplitude of the FRA sigmoid function
		zeta = float(self.parameters['zeta']) # position of the FRA sigmoid function

		# start from biases
		attractiveness = [wRS, wALL, wNOTHING, wBOTTOM, wTOP, wLEFT, wRIGHT, wIN, wOUT]
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
		#if place == -1:
		#	visited = FRA.code2Vector(self.where, 8)
		#	sims1 = [0] + [FRA.sim_consist(visited, x) for x in self.regions]
		#	overlap = FRA.code2Vector(self.joint, 8)
		#	sims2 = [0] + [FRA.sim_consist(overlap, x) for x in self.complements]
		#	sims2[0] = 0 # ALL's complement, NOTHING, does not repel to ALL
		#	FRAsims = np.add(sims1, sims2)
		#	attractiveness = np.add(attractiveness, [delta * FRA.sigmoid(x, epsilon, zeta) for x in FRAsims])

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
		Num_Loc = rounds = self.gameParameters[2] # number of rounds (usually 60)

		# Create players
		Players = []
		for k in range(0, Pl):
			Players.append(player(False, "", 0, [], [], 0, False, int(uniform(0, 1000000)), self.modelParameters[k]))

		# Start the rounds
		for i in range(0, rounds):
			# Playing round i

			#Initializing players for round
			for pl in Players:
				pl.decision = ""
				pl.where = []
				pl.ready = False
				pl.score = 0
				pl.accuracy = False
			
			# Determine players' chosen region
			chosen_strategies = []
			for k in range(0, Pl):
				chosen = Players[k].choice
				if chosen == 0:
					n = randint(2, Num_Loc * Num_Loc - 2)
					chosen_strategies.append(list(np.random.choice(Num_Loc * Num_Loc, n, replace=False)))
				else:
					chosen_strategies.append(Players[k].strategies[chosen - 1])
				chosen_strategies[k] = self.shake(chosen_strategies[k])

			# Start iterations
			for j in range(0, Num_Loc * Num_Loc + 1):
				# Running iteration j
				for k in range(0, Pl):
					# If the other player did not say Present, and current player is not ready, then...
					if not Players[k].ready:
						# ...look at the location determined by the strategy
						# Check if strategy is not over...
						if j<len(chosen_strategies[k]):
							search_place = chosen_strategies[k][j]
							Players[k].where.append(search_place)
						# Otherwise, say Absent
						else:
							# The strategy is over, so guess Absent
							Players[k].decision = "Absent"
							Players[k].ready = True
					# Check if both players are ready. If so, stop search
					elif Players[1-k].ready == True:
						break
				else:
				# Not finished yet
					continue
				break

			# Get results and store data in dataframe (returns players with updated scores)
			Players = self.round2dataframe(Players, i+1, TO_FILE)

			# Players determine their next strategies
			for k in range(0,Pl):
				Players[k].make_decision()
	
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
			colr = list(range(self.gameParameters[2]))
			for l in range(0, Num_Loc * Num_Loc):
				if l in Players[k].where:
					row_of_data[colr[l]] = [1]
				else:
					row_of_data[colr[l]] = [0]
			row_of_data['Score'] = [Players[k].score]
			row_of_data['Strategy'] = [Players[k].choice]
			# Add data to dataFrame
			dfAux = pd.DataFrame.from_dict(row_of_data)
			# Keeping the order of columns
			dfAux = dfAux[['Group','Round','Player','Decision']+colr+['Score','Strategy']]

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