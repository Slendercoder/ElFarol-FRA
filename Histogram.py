print("loading packages...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../Data/humans_only_absent.csv", index_col=False)

fig2, axes2 = plt.subplots()

dyads_LR = ['419-723', '435-261', '469-569', '637-838', '648-175', \
            '656-979', '947-704', '948-444']

dyads_AN = ['352-425', '356-137', '416-710', '462-640', '487-811', '505-833', '938-219']

dyads_TD = ['152-727', '251-716', '277-491', '331-863', \
            '354-344', '546-111', '588-564', '768-541', '825-534']

dyads_IO = ['140-615']

Dyads = dyads_LR + dyads_AN + dyads_TD + dyads_IO

contador = 0
converge = []
dict = {}
w = 2
data['GraW'] = data.rolling(window=w)['DLIndex'].mean()
for key, Grp in data.groupby(['Dyad']):
    Players = Grp.Player.unique()
    GGrp = Grp.groupby(['Player']).get_group(Players[0])
    # if GGrp.DLIndex.mean() > 0.77:
    if key in Dyads:
        # Attempting to get the first round where dyad converges
        try:
            c = list(GGrp[GGrp['GraW']>0.995].index)[0]
        except:
            c = GGrp.GraW.idxmax()
        # c = GGrp[GGrp['GraW']>0.995]['index'].tolist()[0]
        # print(c)
        a = data.Round.iloc[c]
        dict[key] = (data.GraW.iloc[c], a - (w - 1))
        converge.append(a - (w - 1))
        contador += 1
#
# print(dict)
print(converge)
plt.hist(converge)
plt.xlabel('Rounds')
plt.ylabel('Frequency')
plt.xlim([0,60])
#
# print("Num. successful dyads", contador)
plt.show()
