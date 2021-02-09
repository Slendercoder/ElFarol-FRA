import numpy as np
import pandas as pd
import FRA

class Measuring :

    def __init__(self, data, Num_Loc, TOLERANCIA) :
        self.data = data
        self.Num_Loc = 8
        self.TOLERANCIA = TOLERANCIA
        self.CONTADOR = 0
        self.cols = ['a' + str(i) + str(j) \
        for i in range(1, Num_Loc + 1) \
        for j in range(1, Num_Loc + 1) \
        ]
        regionsCoded = ['abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789;:', # ALL
                  '', # NOTHING
                  'GHIJKLMNOPQRSTUVWXYZ0123456789;:', # BOTTOM
                  'abcdefghijklmnopqrstuvwxyzABCDEF', # TOP
                  'abcdijklqrstyzABGHIJOPQRWXYZ4567', # LEFT
                  'efghmnopuvwxCDEFKLMNSTUV012389;:', # RIGHT
                  'jklmnorstuvwzABCDEHIJKLMPQRSTUXYZ012', # IN
                  'abcdefghipqxyFGNOVW3456789;:' # OUT
                  ]
        self.regions = [FRA.code2Vector(FRA.lettercode2Strategy(x, Num_Loc), Num_Loc) for x in regionsCoded]

    def delete_score(self, x):
        if x['Is_there'] == 'Unicorn_Present':
            return np.nan
        else:
            return x['Score']

    def get_measures(self, lista):

        '''List of measures:
        1: Keep only absent
        2: Classify regions
        3: Find max similarity'''

        data = self.data
        print("Sorting by Dyad, Player, Round...")
        data = data.sort_values(['Dyad', 'Player', 'Round'], ascending=[True, True, True]).reset_index(drop=True)
        data['Is_there_LEAD'] = data.groupby(['Dyad', 'Player'])['Is_there'].transform('shift', periods=-1)

        # --------------------------------------------------
        # Obtaining measures from players' performance
        # --------------------------------------------------
        data['Score'] = data['Score'].astype(int)
        print("Finding accumulated score...")
        data['Ac_Score'] = data.sort_values(['Dyad','Player']).groupby('Player')['Score'].cumsum()
        print("Finding the initial lag variables...")

        if '1' in lista:
            # --------------------------------------------------
            # Keeping only rounds with unicorn absent
            # --------------------------------------------------
            print("Keeping only rounds with unicorn absent...")
            # Obtaining nans from unicorn_present blocks
            data['Score'] = data.apply(self.delete_score, axis=1)
            data['ScoreLEAD'] = data.groupby(['Dyad', 'Player'])\
                                        ['Score'].transform('shift', -1)
            data = pd.DataFrame(data.groupby('Is_there').get_group('Unicorn_Absent'))
            # print('List of blocks\n', data[['Round', 'Is_there', 'Score', 'ScoreLEAD']][0:20])

        # --------------------------------------------------
        # Obtaining final measures from players' performance
        # --------------------------------------------------
        cols1 = self.cols
        print("Finding Size_visited...")
        data['Size_visited'] = data[cols1].sum(axis=1)
        print("Finding consistency...")
        data['Vector'] = data.apply(lambda x: np.array(x[cols1]), axis=1)
        data['VectorLAG1'] = data.groupby(['Dyad', 'Player'])['Vector'].transform('shift', 1)
        data['Consistency'] = data.apply(lambda x: FRA.sim_consist(x['Vector'], x['VectorLAG1']), axis=1)
        del data['VectorLAG1']
        print("Finding difference in consistency and Total_visited_dyad...")
        cols = ['Dyad','Player','Consistency','Round','Joint', 'Size_visited', 'Vector']
        total = {}
        dif_cons = {}
        joints = {}
        for key, grp in data[cols].groupby(['Dyad']):
        	Players = grp.Player.unique()
        	Grp_player = grp.groupby(['Player'])
        	aux1 = pd.DataFrame(Grp_player.get_group(Players[0])).reset_index()
        	aux2 = pd.DataFrame(Grp_player.get_group(Players[1])).reset_index()
        	if len(aux1) != len(aux2):
        	       print("Oops, something went wrong with estimated blocks!")
        	       print("Trying to correct...")
        	       print(list(aux1.Round))
        	       d = aux1[aux1['Round'] == 60].index
        	       print('Index round 60', d)
        	       print(aux1.loc[d]['Round'])
        	       aux1 = aux1.drop(d)
        	       print(list(aux1.Round))

        	errorDebug = "Something wrong with players!\n"
        	errorDebug += "Dyad " + str(key)
        	errorDebug += "\nlen player 1 " + str(len(aux1))
        	errorDebug += "\nlen player 2 " + str(len(aux2))
        	errorDebug += "\nRounds player 1 " + str(list(aux1.Round))
        	errorDebug += "\nRounds player 1 " + str(list(aux2.Round))
        	assert(len(aux1) == len(aux2)), errorDebug
        	aux3 = pd.DataFrame({'Dyad':aux1['Dyad'],\
        	'Round':aux1['Round'],\
        	'C1':aux1['Consistency'],\
        	'C2':aux2['Consistency'],\
        	'v1':aux1['Vector'],\
        	'v2':aux2['Vector'],\
        	'V1':aux1['Size_visited'],\
        	'V2':aux2['Size_visited']})
        	aux3['Joint'] = aux3.apply(lambda x: np.sum(np.multiply(x['v1'],x['v2'])), axis=1)
        	aux3['total_visited'] = aux3.apply(lambda x: x['V1'] + x['V2'] - x['Joint'], axis=1)
        	aux3['Dif_consist'] = aux3.apply(lambda x: np.abs(x['C1'] - x['C2']), axis=1)
        	aux3['Pair'] = aux3.apply(lambda x: tuple([x['Dyad'], x['Round']]), axis=1)
        	joints1 = dict(zip(aux3.Pair, aux3.Joint))
        	joints = {**joints, **joints1}
        	total1 = dict(zip(aux3.Pair, aux3.total_visited))
        	total = {**total, **total1}
        	dif_cons1 = dict(zip(aux3.Pair, aux3.Dif_consist))
        	dif_cons = {**dif_cons, **dif_cons1}

        data['Pair'] = data.apply(lambda x: tuple([x['Dyad'], x['Round']]), axis=1)
        data['Joint'] = data['Pair'].map(joints)
        data['Total_visited_dyad'] = data['Pair'].map(total)
        data['Dif_consist'] = data['Pair'].map(dif_cons)
        del data['Vector']
        del data['Pair']

        # Division of labor Index
        data['DLIndex'] = (data['Total_visited_dyad'] - data['Joint'])/(self.Num_Loc * self.Num_Loc)
        assert(all(data['DLIndex'] >= 0)), str(list(data.loc[data['DLIndex']==0].index))

        if '2' in lista:
            # --------------------------------------------------
            # Classify region per round, per player
            # --------------------------------------------------
            print("Classifying regions...")
            cols1 = self.cols
            data['Category'] = data.apply(lambda x: FRA.classify_region(x[cols1], self.TOLERANCIA), axis=1)
        else:
            print("Trying to obtain classification from simulation...")
            try:
                data['Category'] = data.apply(lambda x: FRA.nameRegion(x['Strategy']), axis=1)
            except:
                print('Data does not seem to come from simulation!')

        if '3' in lista:
            # --------------------------------------------------
            # Finding distance to closest focal region per round, per player
            # --------------------------------------------------
            print("Finding distances to focal regions (please be patient)...")
            data['Similarity'] = data.apply(lambda x: FRA.minDist2Focal(x[self.cols], self.regions), axis=1)

        # --------------------------------------------------
        # Finding the lag and lead variables
        # --------------------------------------------------
        print("Finding the last lag variables...")
        data['Score_LAG1'] = data.groupby(['Dyad', 'Player'])\
                                    ['Score'].transform('shift', 1)
        data['Consistency_LEAD1'] = data.groupby(['Dyad', 'Player'])\
                                    ['Consistency'].transform('shift', -1)
        data['Dif_consist_LAG1'] = data.groupby(['Dyad', 'Player'])\
                                    ['Dif_consist'].transform('shift', 1)
        data['RegionGo'] = data.groupby(['Dyad', 'Player'])\
                                    ['Category'].transform('shift', -1)
        if '1' in lista:
            data = data[data['ScoreLEAD'].notna()]

        if '3' in lista:
            data['Similarity_LAG1'] = data.groupby(['Dyad', 'Player'])\
                                    ['Similarity'].transform('shift', 1)

        return data
