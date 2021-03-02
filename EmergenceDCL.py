from random import choices, uniform, randint
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

	def __init__(self, Decision, Choice, Where, Score, Name, modelParameters):
		self.decision = Decision
		self.choice = Choice
		self.where = Where
		self.score = Score
		self.name = Name
		self.parameters = modelParameters
		self.overcrowded = []

		# 5 players:
		self.regions = [[],[],[],[],[],[],[]] # ALL, NOTHING, ALTERS 1-5
		for i in range(64):
			if i%5==0:
				self.regions[0].append(1)
				self.regions[1].append(0)
				self.regions[2].append(1)
				self.regions[3].append(1)
				self.regions[4].append(1)
				self.regions[5].append(0)
				self.regions[6].append(0)
			elif i%5==1:
				self.regions[0].append(1)
				self.regions[1].append(0)
				self.regions[2].append(1)
				self.regions[3].append(1)
				self.regions[4].append(0)
				self.regions[5].append(0)
				self.regions[6].append(1)
			elif i%5==2:
				self.regions[0].append(1)
				self.regions[1].append(0)
				self.regions[2].append(1)
				self.regions[3].append(0)
				self.regions[4].append(0)
				self.regions[5].append(1)
				self.regions[6].append(1)
			elif i%5==3:
				self.regions[0].append(1)
				self.regions[1].append(0)
				self.regions[2].append(0)
				self.regions[3].append(0)
				self.regions[4].append(1)
				self.regions[5].append(1)
				self.regions[6].append(1)
			elif i%5==4:
				self.regions[0].append(1)
				self.regions[1].append(0)
				self.regions[2].append(0)
				self.regions[3].append(1)
				self.regions[4].append(1)
				self.regions[5].append(1)
				self.regions[6].append(0)
		
		# 2 players:
		self.regions = [[0,1]*30,[1,0]*30]

	def decide(self):
		attractiveness = self.attract2p() #attract/attract2p
		sum = np.sum(attractiveness)
		probabilities = [x/sum for x in attractiveness]
		newChoice = choices(range(len(self.parameters)-5), weights=probabilities)[0]
		self.choice = newChoice

	def attract(self, DEB=False):
		wALL = float(self.parameters['ALL'])
		wNOTHING = float(self.parameters['NOTHING'])
		wALTER1 = float(self.parameters['ALTER1'])
		wALTER2 = float(self.parameters['ALTER2'])
		wALTER3 = float(self.parameters['ALTER3'])
		wALTER4 = float(self.parameters['ALTER4'])
		wALTER5 = float(self.parameters['ALTER5'])
		attractiveness = [wALL, wNOTHING, wALTER1, wALTER2, wALTER3, wALTER4, wALTER5]
		wRS = 1 - np.sum(np.array(attractiveness))
		assert(wRS >= 0), "Incorrect biases! Sum greater than 1"
		attractiveness = [wRS] + attractiveness

		alpha = float(self.parameters['alpha']) # for how much the focal region augments attractiveness
		beta = float(self.parameters['beta']) # amplitude of the WSLS sigmoid function
		gamma = float(self.parameters['gamma']) # position of the WSLS sigmoid function
		delta = float(self.parameters['delta']) # for how much the added FRA similarities augments attractiveness
		epsilon = float(self.parameters['epsilon']) # amplitude of the FRA sigmoid function
		zeta = float(self.parameters['zeta']) # position of the FRA sigmoid function

		# start from biases
		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('Player', self.name)
			print('attractiveness before WS and FRA\n', attactPrint)

		# Adding 'Win Stay'
		if self.choice != 0:
			attractiveness[self.choice] += alpha * FRA.sigmoid(self.score[-1], beta, gamma)

		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('attractiveness with WS\n', attactPrint)

		partial_regions = [r[:len(self.where)] for r in self.regions]
		complements = [[1 - x for x in sublist] for sublist in partial_regions]
		# Adding 'FRA'
		sims1 = [0] + [FRA.distance(self.where, x) for x in partial_regions]
		sims2 = [0] + [FRA.distance(self.overcrowded, x) for x in complements]
		FRAsims = np.add(sims1, sims2)
		attractiveness = np.add(attractiveness, [delta *(1 - FRA.sigmoid(x, epsilon, zeta)) for x in FRAsims])

		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('attractiveness with FRA\n', attactPrint)

		return attractiveness

	def attract2p(self, DEB=False):
		wALTER1 = float(self.parameters['ALTER1'])
		wALTER2 = float(self.parameters['ALTER2'])
		attractiveness = [wALTER1, wALTER2]
		wRS = 1 - np.sum(np.array(attractiveness))
		assert(wRS >= 0), "Incorrect biases! Sum greater than 1"
		attractiveness = [wRS] + attractiveness

		alpha = float(self.parameters['alpha']) # for how much the focal region augments attractiveness
		beta = float(self.parameters['beta']) # amplitude of the WSLS sigmoid function
		gamma = float(self.parameters['gamma']) # position of the WSLS sigmoid function
		delta = float(self.parameters['delta']) # for how much the added FRA similarities augments attractiveness
		epsilon = float(self.parameters['epsilon']) # amplitude of the FRA sigmoid function
		zeta = float(self.parameters['zeta']) # position of the FRA sigmoid function

		# start from biases
		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('Player', self.name)
			print('attractiveness before WS and FRA\n', attactPrint)

		# Adding 'Win Stay'
		if self.choice != 0:
			attractiveness[self.choice] += alpha * FRA.sigmoid(self.score[-1], beta, gamma)

		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('attractiveness with WS\n', attactPrint)

		partial_regions = [r[:len(self.where)] for r in self.regions]
		complements = [[1 - x for x in sublist] for sublist in partial_regions]
		# Adding 'FRA'
		sims1 = [0] + [FRA.distance(self.where, x) for x in partial_regions]
		sims2 = [0] + [FRA.distance(self.overcrowded, x) for x in complements]
		FRAsims = np.add(sims1, sims2)
		attractiveness = np.add(attractiveness, [delta * (1 - FRA.sigmoid(x, epsilon, zeta)) for x in FRAsims])

		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('attractiveness with FRA\n', attactPrint)

		return attractiveness

# Define Experiment Object
class Experiment :
	'''Object defining the experiment and simulation'''

	def __init__(self, gameParameters, modelParameters):
		assert(len(gameParameters) == 4), "Game parameters incorrect length!"
		self.gameParameters = gameParameters
		self.modelParameters = modelParameters
		# Create data frame
		cols = ['Group', 'Round', 'Player','Decision','Score', 'Strategy']
		self.df = pd.DataFrame(columns=cols)

	def run_group(self, TO_FILE=True):
		p = self.gameParameters[0] # threshold (usually 0.6)
		Pl = self.gameParameters[1] # number of players (usually 5)
		rounds = self.gameParameters[2] # number of rounds (usually 60)

		# Create players
		Players = []
		for k in range(0, Pl):
			Players.append(player(0, 0, [], [], int(uniform(0, 1000000)), self.modelParameters))

		# Start the rounds
		for Num_Loc in range(rounds):
			# Playing round i
			
			# Determine players' chosen region
			chosen_strategies = []
			for k in range(0, Pl):
				chosen = Players[k].choice
				if chosen == 0:
					chosen_strategies.append([randint(0,1) for i in range(Num_Loc+1)])
				else:
					chosen_strategies.append(Players[k].regions[chosen - 1])

			# Player decides whether to assist or not
			for k in range(0, Pl):
				Players[k].decision = chosen_strategies[k][Num_Loc]
				Players[k].where.append(Players[k].decision)

			# Get results and store data in dataframe (returns players with updated scores)
			Players = self.round2dataframe(Players, Num_Loc+1, TO_FILE)

			# Players determine their next strategies
			for k in range(0,Pl):
				Players[k].decide()

	def run_simulation(self):
		iters = self.gameParameters[3] # number of experiments in a set
		for g in range(0, iters):
			#print("****************************")
			#print("Running group no. ", g + 1)
			#print("****************************\n")
			self.run_group()
	
	def round2dataframe(self, Players, round, TO_FILE):
		# Create row of data as dictionary
		row_of_data = {}
		# Create group name
		group = ''
		for pl in Players: group += str(pl.name)[:5]
		# Determine whether bar was overcrowded
		overcrowded = len([p for p in Players if p.decision == 1])/len(Players) > self.gameParameters[0]
		# Save data per player
		for k in range(0, len(Players)):
			Players[k].overcrowded.append(int(overcrowded))
			# Determine individual scores
			if overcrowded:
				# Bar was overcrowded
				if Players[k].decision == 0:
					# Player k didn't go
					Players[k].score.append(0)
				else:
					# Player k's decision is Incorrect
					Players[k].score.append(-1)
			else:
				# Bar was not overcrowded
				if Players[k].decision == 0:
					# Player k didn't go
					Players[k].score.append(0)
				else:
					# Player k's decision is Correct
					Players[k].score.append(1)
			row_of_data['Group'] = [group]
			row_of_data['Round'] = [round]
			row_of_data['Player'] = [Players[k].name]
			row_of_data['Decision'] = [Players[k].decision]
			row_of_data['Score'] = [Players[k].score[-1]]
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