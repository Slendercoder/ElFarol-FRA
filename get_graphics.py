print("loading packages...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sys import argv
import os

data_archivo = "../Data/humans_full.csv"

Num_Loc = 8

Vid_graph = input("Do you want to create graphics for measures? (1=YES/0=NO): ")
Vid_graph = int(Vid_graph)

LaTeX_file = input("Do you want to produce the LaTeX file to show the figures? (1=YES/0=NO): ")
LaTeX_file = int(LaTeX_file)

Vid = input("Do you want to create figures for the videos? (1=YES/0=NO): ")
Vid = int(Vid)

# Opens the file with data from DCL experiment and parses it into data
print("Reading data...")
#data = pd.read_csv(data_archivo, sep='\t', header=0)
data = pd.read_csv(data_archivo, index_col=False)
#data = pd.read_csv(data_archivo, index_col=False)
print("Data red!")
# print data[:3]

# Find the min value of score
minimum_score = data['Score'].min()
maximum_score = data['Score'].max()

# Find the number of rounds
Num_Rondas = data.Round.unique().max()
print("Numero de rondas: ", Num_Rondas)
# This is for the graphics with visited locations
step = 1. / Num_Loc

# Prepare dataframe with NaN for missing values
auxDF = pd.DataFrame(pd.Series(range(1,60)))
auxDF.columns = ['ronda']
xs = auxDF['ronda'].tolist()
xs = np.array(xs)

directorio_graficas = 'Graficas/'


# directorios = ['../Graficas/score', '../Graficas/ac_score', \
#     '../Graficas/consistency/', '../Graficas/dist_path', \
#     '../Graficas/dist_comp', '../Graficas/vis_loc', \
#     '../Graficas/Demographic', '../Graficas/Norm_Score',\
#     '../Graficas/fairness', '../Graficas/dlindex']

# directorios = ['/Graficas', '/Graficas/score', '/Graficas/ac_score', \
#     '/Graficas/consistency/', '/Graficas/dist_path', \
#     '/Graficas/dist_comp', '/Graficas/vis_loc']
#
# print("Verifying paths...")
#
# for d in directorios:
#     try:
#         os.makedirs(d)
#         print("Creating " + d)
#     except OSError:
#         if not os.path.isdir(d):
#             raise

if Vid_graph==1:

    # Modifica el tamanho de letra de las legendas en las graficas
    plt.rc('legend', fontsize=16)

    # Produce graphics
    print("Producing graphics...")
    for Key, grp in data.groupby(['Dyad']):

        # figs for Score
        print("Preparando figuras para score por ronda...")
        fig, axes = plt.subplots(1,2)
        for a in axes:
            a.set_xlabel('Rounds (unicorn absent)', fontsize = 12)
            # a.set_ylabel('Score', fontsize = 14)
            a.set_ylim([minimum_score-2, maximum_score+2])
        axes[0].set_ylabel('Score', fontsize = 14)
        axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position("right")

        print("Preparando figuras para accumulated score por ronda...")
        # figs for Accumulated Score
        fig1, axes1 = plt.subplots(1,2)
        for a in axes1:
            a.set_xlabel('Rounds (unicorn absent)', fontsize = 12)
            # a.set_ylabel('Accumulated Score', fontsize = 14)
        axes1[0].set_ylabel('Accumulated Score', fontsize = 14)
        axes1[1].yaxis.tick_right()
        axes1[1].yaxis.set_label_position("right")

        print("Preparando figuras DLIndex...")
        # figs for DLIndex
        fig2, axes2 = plt.subplots()
        axes2.set_xlabel('Rounds (unicorn absent)', fontsize = 14)
        axes2.set_ylabel('DLIndex', fontsize = 14)
        axes2.set_ylim(-0.1, 1.1)

        print("Preparando figuras consistency paths t and t+1 per round...")
        # figs for Consistency
        fig3, axes3 = plt.subplots(1,2)
        for a in axes3:
            a.set_xlabel('Rounds (unicorn absent)', fontsize = 14)
            # a.set_ylabel('Consistency', fontsize = 14)
            a.set_ylim(-0.1, 1.1)
        axes3[0].set_ylabel('Consistency', fontsize = 14)
        axes3[1].yaxis.tick_right()
        axes3[1].yaxis.set_label_position("right")

        print("Preparando figuras locaciones visitadas...")
        # figs for visited locations
        fig4, axes4 = plt.subplots(1,2)
        for a in axes4:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)

        # print("Preparando figuras fairness y dif_consist per round...")
        # # figs for fairness and dlindex
        # fig5, axes5 = plt.subplots(1,2)
        # axes5[0].set_xlabel('Rounds (unicorn absent)', fontsize = 18)
        # axes5[0].set_ylabel('Fairness', fontsize = 18)
        # axes5[0].set_ylim(-0.1, 1.1)
        # axes5[1].set_xlabel('Rounds (unicorn absent)', fontsize = 18)
        # axes5[1].set_ylabel('Dif_consist', fontsize = 18)
        # axes5[1].set_ylim(-0.1, 1.1)
        # axes5[1].yaxis.tick_right()
        # axes5[1].yaxis.set_label_position("right")

        col1 = ['a' + str(a + 1) + str(b + 1) for a in range(0, Num_Loc) for b in range(0, Num_Loc)]
        # print col1
        step = 1. / Num_Loc

        # Plot per player
        i = 0
        for key, Grp in grp.groupby(['Player']):

            fig.suptitle('Dyad: ' + str(Key))
            # Plot Score
            auxDF['Score'] = auxDF['ronda'].apply(lambda x: x*0)
            d = pd.Series(Grp.Score.values,index=Grp.Round).to_dict()
            auxDF['Score'] = auxDF['ronda'].map(d)
            series1 = auxDF['Score'].tolist()
            series1 = np.array(series1)
            s1mask = np.isfinite(series1)
            axes[i].plot(xs[s1mask], series1[s1mask], \
                     linestyle='-', color='black')
            axes[i].set_title('Player: ' + str(key))
            # Grp['Score'].plot(use_index=False, x=Grp['Round'], ax=axes[i], \
            #                     title='Player: ' + str(key))
            print("Score of player " + str(Key) + " dibujado")

            # Plot Accumulated Score
            fig1.suptitle('Dyad: ' + str(Key))
            auxDF['Ac_Score'] = auxDF['ronda'].apply(lambda x: x*0)
            d = pd.Series(Grp.Ac_Score.values,index=Grp.Round).to_dict()
            auxDF['Ac_Score'] = auxDF['ronda'].map(d)
            series1 = auxDF['Ac_Score'].tolist()
            series1 = np.array(series1)
            s1mask = np.isfinite(series1)
            axes1[i].plot(xs[s1mask], series1[s1mask], \
                     linestyle='-', color='black')
            axes1[i].set_title('Player: ' + str(key))
            # Grp['Ac_Score'].plot(use_index=False, x=Grp['Round'], ax=axes1[i], \
            #                         title='Player: ' + str(key))
            print("Accumulated Score of player " + str(Key) + " dibujado")

            # Plot Consistency
            fig3.suptitle('Dyad: ' + str(Key))
            auxDF['Consistency'] = auxDF['ronda'].apply(lambda x: x*0)
            d = pd.Series(Grp.Consistency.values,index=Grp.Round).to_dict()
            auxDF['Consistency'] = auxDF['ronda'].map(d)
            series1 = auxDF['Consistency'].tolist()
            series1 = np.array(series1)
            s1mask = np.isfinite(series1)
            axes3[i].plot(xs[s1mask], series1[s1mask], \
                     linestyle='-', color='black')
            axes3[i].set_title('Player: ' + str(key))
            # Grp['Consistency'].plot(use_index=False, x=Grp['Round'], ax=axes3[i], \
            #                         title='Player: ' + str(key))
            print("Consistency of player " + str(Key) + " dibujado")

            # Plot visited locations
            ejemp = [Grp[c].sum() for c in col1]
            # m = 90
            m = Num_Rondas
            tangulos = []
            for j in range(0, len(ejemp)):
                x = int(j) % Num_Loc
                y = (int(j) - x) / Num_Loc
                # print "x: " + str(x)
                # print "y: " + str(y)
                by_x = x * step
                by_y = 1 - (y + 1) * step
                # print "by_x: " + str(by_x)
                # print "by_y: " + str(by_y)
                tangulos.append(patches.Rectangle(*[(by_x, by_y), step, step],\
                    facecolor="black", alpha=float(ejemp[j])/m))
            for t in tangulos:
                axes4[i].add_patch(t)
            axes4[i].set_title('Player ' + str(key))
            i += 1
            print("Visited locations de la pareja " + str(Key) + " dibujado")

        # Plot DLIndex
        Player = grp.Player.unique()
        data_pl = grp.groupby('Player').get_group(Player[0])
        # ... for DLIndex
        auxDF['DLIndex'] = [0]*len(auxDF['ronda'])
        d = pd.Series(data_pl.DLIndex.values,index=data_pl.Round).to_dict()
        auxDF['DLIndex'] = auxDF['ronda'].map(d)
        series1 = auxDF['DLIndex'].tolist()
        series1 = np.array(series1)
        s1mask = np.isfinite(series1)
        axes2.plot(xs[s1mask], series1[s1mask], \
                 linestyle='-', color='black')
        axes2.set_title('Player: ' + str(key))
        # data_pl['DLIndex'].plot(use_index=False, x=data_pl['Round'], ax=axes2, \
        #                         title='Dyad: ' + str(Key))
        print("DLIndex de la pareja " + str(Key) + " dibujado")

        # # Plot Fairness
        # Player = grp.Player.unique()
        # # print "Dyad " + str(Key) + " has players: "
        # # print Player
        # data_pl = grp.groupby('Player').get_group(Player[0])
        # fig5.suptitle('Dyad: ' + str(Key))
        # data_pl['Fairness'].plot(use_index=False, x=data_pl['Round'], ax=axes5[0])
        # print("Fairness de la pareja " + str(Key) + " dibujado")
        # data_pl['Dif_consist'].plot( \
        #                             use_index=False, x=data_pl['Round'], ax=axes5[1])
        # print "Dif_consist de la pareja " + str(Key) + " dibujado"

        print("Verifying paths for dyad...")
        d = directorio_graficas + str(Key)
        try:
            os.makedirs(d)
            print("Creating " + d)
        except OSError:
            if not os.path.isdir(d):
                raise

        fig.savefig(directorio_graficas + str(Key) + '/score.png')
        fig1.savefig(directorio_graficas + str(Key) + '/ac_score.png')
        fig2.savefig(directorio_graficas + str(Key) + '/DLIndex.png')
        fig3.savefig(directorio_graficas + str(Key) + '/Consistency.png')
        fig4.savefig(directorio_graficas + str(Key) + '/visited_locations.png')
        # fig5.savefig(directorio_graficas + str(Key) + '/fairness_difconsist.png')
        plt.close(fig)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        # plt.close(fig5)

if LaTeX_file == 1:
    # --------------------------------------------------
    # Produce LaTeX file
    # --------------------------------------------------
    print("Producing LaTeX file...")

    fl = open('graphics.tex', 'w')

    fl.write('% Este es el archivo consolidado de graficas\n')
    fl.write('\n \documentclass{article}\n')
    fl.write('\\usepackage[top=1cm,bottom=1cm,left=1cm,right=1cm]{geometry}\n')
    fl.write('\\usepackage{graphicx}\n')
    fl.write('\n \\begin{document}\n')

    Dyads = data.Dyad.unique()

    for Key in Dyads:
        fl.write('\hspace*{-1.5cm}\\begin{tabular}{cc}\n')
        fl.write('\includegraphics[scale=0.5]{Graficas/' + str(Key) + '/score.png} &')
        fl.write('\includegraphics[scale=0.5]{Graficas/' + str(Key) + '/ac_score.png} \cr \n')
        fl.write('\includegraphics[scale=0.5]{Graficas/' + str(Key) + '/DLIndex.png} &')
        fl.write('\includegraphics[scale=0.5]{Graficas/' + str(Key) + '/Consistency.png} \cr \n')
        fl.write('\includegraphics[scale=0.5]{Graficas/' + str(Key) + '/visited_locations.png} & \n')
        # fl.write('\includegraphics[scale=0.5]{Graficas/' + str(Key) + '/fairness_difconsist.png} \cr \n')
        fl.write('\end{tabular}\n')
        fl.write('\n \pagebreak\n')

    fl.write('\n \end{document}\n')

    fl.close()
    print('LaTeX file produced!')
    print('You can run\n\t> pdflatex graphics.tex\n to obtain the graphics in a single pdf file')

if Vid == 1:
    # --------------------------------------------
    # Creando figuras para los videos
    # --------------------------------------------
    for Key, grp in data.groupby(['Dyad']):
        Players = grp.Player.unique()
        print("The players in dyad " + str(Key) + " are: " + str(Players))

        print("Verifying paths for dyad...")
        directorio = directorio_graficas + str(Key) + '/Videos'
        try:
            os.makedirs(directorio)
            print("Creating " + directorio)
        except OSError:
            if not os.path.isdir(directorio):
                raise

        contador = 1
        # print "contador: ", contador
        for ronda, valores in grp.groupby(['Round']):
            print("Preparando figuras locaciones visitadas ronda " + str(ronda))

            # figs for visited locations
            fig4, axes4 = plt.subplots(1,2)
            for a in axes4:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)

            # Plot joint tiles
            Grp_player = valores.groupby(['Player'])
            Player = grp.Player.unique()
            aux1 = pd.DataFrame(Grp_player.get_group(Players[0]))
            # print aux1
            aux2 = pd.DataFrame(Grp_player.get_group(Players[1]))
            # print aux2

            # Plot visited locations per player
            play = 0
            for key, casillas in valores.groupby(['Player']):
                # print "Trabajando con el jugador " + str(key)
                tangulos = []
                for j in range(0, Num_Loc * Num_Loc):
                    x = int(int(j) % Num_Loc)
                    y = int((int(j) - x) / Num_Loc)
                    # print "x: " + str(x + 1)
                    # print "y: " + str(y + 1)
                    colA = "a" + str(x + 1) + str(y + 1)
                    # print colA
                    by_x = x * step
                    by_y = 1 - (y + 1) * step
                #     # print "by_x: " + str(by_x)
                #     # print "by_y: " + str(by_y)
                    if (list(aux1[colA].unique())[0] == 1) and (list(aux2[colA].unique())[0] == 1):
                        tangulos.append(patches.Rectangle(*[(by_y, by_x), step, step],\
                            facecolor="red"))
                    else:
                        if Players.tolist().index(key) == 0:
                            if list(aux1[colA].unique())[0] == 1:
                                tangulos.append(patches.Rectangle(*[(by_y, by_x), step, step],\
                                    facecolor="black", alpha=1))
                        elif Players.tolist().index(key) == 1:
                            if list(aux2[colA].unique())[0] == 1:
                                tangulos.append(patches.Rectangle(*[(by_y, by_x), step, step],\
                                    facecolor="black", alpha=1))

                # print "Dibujando recorrido jugador ", play
                for t in tangulos:
                    axes4[play].add_patch(t)
                axes4[play].set_title('Player ' + str(key))
                play += 1

            archivo = directorio_graficas + str(Key) + '/Videos/snapshot' + str(contador).zfill(3) + '.png'
            # print "archivo: ", archivo
            fig4.savefig(archivo)
            plt.close(fig4)
            contador += 1

print("Done!")
