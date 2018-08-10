"""

Copyright (c) 2018, Silvia Pagliarini, Xavier Hinaut, Arthur Leblois
https://github.com/spagliarini

Mnemosyne team, Inria, Bordeaux, France
https://team.inria.fr/mnemosyne/fr/

Distributed under the BSD-2-Clause License

PLOT: comparison between different normalizations
(Fig. 4)

"""

import os
import numpy as np 
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}

os.chdir('C://Users//Mnemosyne//Documents//Python Scripts//InverseModelBirdsong//results//IMsimple_model//AllNormalizations//NormalizedAN')   #or NormalizedMN 

#load distance files
err_wn=np.load('Distance_simple.npy')
err_2avrg=np.load('Distance_2avrg.npy')
err_2norm=np.load('Distance_2norm.npy')
err_3avrg=np.load('Distance_3avrg.npy')
err_3norm=np.load('Distance_3norm.npy')
err_4avrg=np.load('Distance_4avrg.npy')
err_4norm=np.load('Distance_4norm.npy')
err_5avrg=np.load('Distance_5avrg.npy')
err_5norm=np.load('Distance_5norm.npy')

#definition of the end_time
end_time=err_wn.shape[1]

#smooth average distance over all the simulations
mean_err_wn=np.zeros((end_time,1))
mean_err_2avrg=np.zeros((end_time,1))
mean_err_2norm=np.zeros((end_time,1))
mean_err_3avrg=np.zeros((end_time,1))
mean_err_3norm=np.zeros((end_time,1))
mean_err_4avrg=np.zeros((end_time,1))
mean_err_4norm=np.zeros((end_time,1))
mean_err_5avrg=np.zeros((end_time,1))
mean_err_5norm=np.zeros((end_time,1))
for k in range(0,end_time):
    mean_err_wn[k]=np.mean(err_wn[:,k])
    mean_err_2avrg[k]=np.mean(err_2avrg[:,k])
    mean_err_2norm[k]=np.mean(err_2norm[:,k])
    mean_err_3avrg[k]=np.mean(err_3avrg[:,k])
    mean_err_3norm[k]=np.mean(err_3norm[:,k])
    mean_err_4avrg[k]=np.mean(err_4avrg[:,k])
    mean_err_4norm[k]=np.mean(err_4norm[:,k])
    mean_err_5avrg[k]=np.mean(err_5avrg[:,k])
    mean_err_5norm[k]=np.mean(err_5norm[:,k])

#Model without any normalization applied
x_time=np.linspace(0,end_time-1,end_time)

plt.figure()
ax=plt.subplot() 
plt.plot(x_time,mean_err_wn,'royalblue',label='Simple inverse model')
plt.axis([0, end_time, 0, np.max(mean_err_wn)+2])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.legend(loc='upper left', fontsize=8)
plt.xlabel('Time (in number of time steps)', **csfont, fontsize=8)
plt.ylabel('Evolution of the distance', **csfont, fontsize=8)
plt.xticks([0,end_time/2,end_time])
plt.savefig('IMsimple.pdf')

#Cfr ALL convergent normalizations 
plt.figure()
ax=plt.subplot() 
norm2avrg,=plt.plot(x_time,mean_err_2avrg,'steelblue',label='Normalization (1a)')
norm2norm,=plt.plot(x_time,mean_err_2norm,'steelblue',label='Normalization (1b)', linestyle='--')
norm3avrg,=plt.plot(x_time,mean_err_3avrg,'forestgreen', alpha=.70,label='Normalization (2a)')
norm3norm,=plt.plot(x_time,mean_err_3norm,'forestgreen', alpha=.70,label='Normalization (2b)', linestyle='--')
norm4avrg,=plt.plot(x_time,mean_err_4avrg,'orangered',label='Normalization (3a)')
norm4norm,=plt.plot(x_time,mean_err_4norm,'orangered',label='Normalization (3b)', linestyle='--')
leg = plt.legend(loc='upper right', fontsize=8, ncol=1, shadow=True, fancybox=True)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.yscale('log')
plt.xlabel('Time (in number of time steps)', **csfont, fontsize=8)
plt.ylabel('Evolution of the distance', **csfont, fontsize=8)
plt.xticks([0,end_time/2,end_time]) 
plt.savefig('IMsimple_normalization234_AN.pdf')

#Divergent normalization 
plt.figure()
ax=plt.subplot() 
classic,=plt.plot(x_time,mean_err_wn,'royalblue',label='Classic inverse model')
norm5avrg,=plt.plot(x_time,mean_err_5norm, 'k',label='Normalization (4a)')
norm5norm,=plt.plot(x_time,mean_err_5norm,'orange',label='Normalization (4a)', linestyle='--')
leg = plt.legend(loc='lower right', fontsize=8, ncol=1, shadow=True, fancybox=True)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.xlabel('Time (in number of time steps)', fontsize=15)
plt.ylabel('Evolution of the distance', fontsize=15)
plt.xticks([0,end_time/2,end_time])
plt.savefig('IMsimple_normalization5_MN.pdf')

#auditory VS motor neurons
os.chdir('C://Users//Mnemosyne//Documents//Python Scripts//InverseModelBirdsong//results//IMsimple_model//AllNormalizations') 

mean_err_4avrg_AN=np.zeros((end_time,1))
mean_err_4avrg_MN=np.zeros((end_time,1))
mean_err_4norm_AN=np.zeros((end_time,1))
mean_err_4norm_MN=np.zeros((end_time,1))
err_4avrg_AN=np.load('Distance_4avrgAN.npy')
err_4avrg_MN=np.load('Distance_4avrgMN.npy')
err_4norm_AN=np.load('Distance_4normAN.npy')
err_4norm_MN=np.load('Distance_4normMN.npy')
for k in range(0,end_time):
    mean_err_4avrg_AN[k]=np.mean(err_4avrg_AN[:,k])
    mean_err_4avrg_MN[k]=np.mean(err_4avrg_MN[:,k])
    mean_err_4norm_AN[k]=np.mean(err_4norm_AN[:,k])
    mean_err_4norm_MN[k]=np.mean(err_4norm_MN[:,k])

plt.figure()
ax=plt.subplot() 
avrgAN=plt.plot(x_time,mean_err_4avrg_AN[:end_time],'orangered',label='Normalization (3a) wrt auditory neurons')
avrgMN,=plt.plot(x_time,mean_err_4avrg_MN[:end_time],'darkred',label='Normalization (3a) wrt motor neurons')
normAN=plt.plot(x_time,mean_err_4norm_AN[:end_time],'orangered',label='Normalization (3a) wrt auditory neurons', linestyle='--')
normMN,=plt.plot(x_time,mean_err_4norm_MN[:end_time],'darkred',label='Normalization (3a) wrt motor neurons', linestyle='--')
leg = plt.legend(loc='upper right', fontsize=6, ncol=1, shadow=True, fancybox=True)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.yscale('log')
plt.xlabel('Time (in number of time steps)', **csfont, fontsize=8)
plt.ylabel('Evolution of the distance',**csfont, fontsize=8)
plt.xticks([0,end_time/2,end_time])
plt.savefig('IMsimple_normalization4_ANvsMN.pdf')
