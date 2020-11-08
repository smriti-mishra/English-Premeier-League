# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:07:18 2018

@author: Versha Mom
"""

# 'Converting teams into indexed numbers and change [W,L,D] to numbers
# 'Trying to generate features out of the file we got and try to create classif
# 'Apply a Machine learning Algorithm

#def get_all_names():
#    all_teams = df['HomeTeam'].values.tolist() + df['AwayTeam'].values.tolist()
#    return sorted(list(set(all_teams)))

import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam

Training_Set = pandas.read_csv('outfile1.csv')
Target_Set = pandas.read_csv('outfile2.csv')
#Training_Set = Training_Set[Training_Set.FTR != 'D']
#Target_Set = Target_Set[Target_Set.FTR != 'D']

# construct Poisson  for each mean goals value

poisson_pred = np.column_stack([[poisson.pmf(i, Training_Set.mean()[j]) for i in range(8)] for j in range(2)])

# plot histogram of actual goals
plt.hist(Training_Set[['FTHG', 'FTAG']].values, range(9), 
         alpha=0.7, label=['Home', 'Away'],normed=True, color=["#FFA07A", "#20B2AA"])
# add lines for the Poisson distributions
pois1, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0],
                  linestyle='-', marker='o',label="Home", color = '#CD5C5C')
pois2, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1],
                  linestyle='-', marker='o',label="Away", color = '#006400')

leg=plt.legend(loc='upper right', fontsize=13, ncol=2)
leg.set_title("Poisson           Actual        ", prop = {'size':'14', 'weight':'bold'})

plt.xticks([i-0.5 for i in range(1,9)],[i for i in range(9)])
plt.xlabel("Goals per Match",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Number of Goals per Match (EPL Season 02/03 - 13/14 )",size=14,fontweight='bold')
plt.ylim([-0.004, 0.4])
plt.tight_layout()
plt.show()

skellam_pred = [skellam.pmf(i, Training_Set.mean()[0], Training_Set.mean()[1]) for i in range(-6,9)]
plt.hist(Training_Set[['FTHG']].values - Training_Set[['FTAG']].values, range(-6,9), 
         alpha=0.7, label='Actual',normed=True)
plt.plot([i+0.5 for i in range(-6,9)], skellam_pred,
                  linestyle='-', marker='o',label="Skellam", color = '#CD5C5C')
plt.legend(loc='upper right', fontsize=13)
plt.xticks([i+0.5 for i in range(-6,9)],[i for i in range(-6,9)])
plt.xlabel("Home Goals - Away Goals",size=13)
plt.ylabel("Proportion of Matches",size=13)
plt.title("Difference in Goals Scored (Home Team vs Away Team)",size=14,fontweight='bold')
plt.ylim([-0.004, 0.32])
plt.tight_layout()
plt.show()

fig,(ax1,ax2) = plt.subplots(2, 1)


chel_home = Training_Set[Training_Set['HomeTeam']=='Chelsea'][['FTHG']].apply(pandas.value_counts,normalize=True)
chel_home_pois = [poisson.pmf(i,np.sum(np.multiply(chel_home.values.T,chel_home.index.T),axis=1)[0]) for i in range(9)]
sun_home = Training_Set[Training_Set['HomeTeam']=='Sunderland'][['FTHG']].apply(pandas.value_counts,normalize=True)
sun_home_pois = [poisson.pmf(i,np.sum(np.multiply(sun_home.values.T,sun_home.index.T),axis=1)[0]) for i in range(9)]

chel_away = Training_Set[Training_Set['AwayTeam']=='Chelsea'][['FTAG']].apply(pandas.value_counts,normalize=True)
chel_away_pois = [poisson.pmf(i,np.sum(np.multiply(chel_away.values.T,chel_away.index.T),axis=1)[0]) for i in range(9)]
sun_away = Training_Set[Training_Set['AwayTeam']=='Sunderland'][['FTAG']].apply(pandas.value_counts,normalize=True)
sun_away_pois = [poisson.pmf(i,np.sum(np.multiply(sun_away.values.T,sun_away.index.T),axis=1)[0]) for i in range(9)]

Chel_Hmval = [chel_home.values[i][0] for i in range(9)]
Sun_Hmval =  [sun_home.values[i][0] for i in range(6)]
ax1.bar(chel_home.index-0.4,Chel_Hmval,width=0.4,color="#034694",label="Chelsea")
ax1.bar(sun_home.index,Sun_Hmval,width=0.4,color="#EB172B",label="Sunderland")
pois1, = ax1.plot([i for i in range(9)], chel_home_pois,linestyle='-', marker='o',label="Chelsea", color = "#0a7bff")
pois1, = ax1.plot([i for i in range(9)], sun_home_pois,
linestyle='-', marker='o',label="Sunderland", color = "#ff7c89")
                  
leg=ax1.legend(loc='upper right', fontsize=12, ncol=2)
leg.set_title("Poisson                 Actual                ", prop = {'size':'14', 'weight':'bold'})
ax1.set_xlim([-0.5,7.5])
ax1.set_ylim([-0.01,0.65])
ax1.set_xticklabels([])
# mimicing the facet plots in ggplot2 with a bit of a hack
ax1.text(7.65, 0.585, '                Home                ', rotation=-90,
        bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})
ax2.text(7.65, 0.585, '                Away                ', rotation=-90,
        bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})

Chel_Awval = [chel_away.values[i][0] for i in range(len(chel_away.values))]
Sun_Awval =  [sun_away.values[i][0] for i in range(len(sun_away.values))]


ax2.bar(chel_away.index-0.4,Chel_Awval,width=0.4,color="#034694",label="Chelsea")
ax2.bar(sun_away.index,Sun_Awval,width=0.4,color="#EB172B",label="Sunderland")
pois1, = ax2.plot([i for i in range(len(chel_away_pois))], chel_away_pois,linestyle='-', marker='o',label="Chelsea", color = "#0a7bff")
pois1, = ax2.plot([i for i in range(len(sun_away_pois))], sun_away_pois,linestyle='-', marker='o',label="Sunderland", color = "#ff7c89")
ax2.set_xlim([-0.5,7.5])
ax2.set_ylim([-0.01,0.65])
ax1.set_title("Number of Goals per Match (EPL 2013/14 Season)",size=14,fontweight='bold')
ax2.set_xlabel("Goals per Match",size=13)
ax2.text(-1.15, 0.9, 'Proportion of Matches', rotation=90, size=13)
plt.tight_layout()
plt.show()                          
