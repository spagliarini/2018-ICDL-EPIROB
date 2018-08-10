"""

HAHNLOSER & GANGULI MODEL - 
with nonlinear sensory response

Copyright (c) 2018, Silvia Pagliarini, Xavier Hinaut, Arthur Leblois
https://github.com/spagliarini

Mnemosyne team, Inria, Bordeaux, France
https://team.inria.fr/mnemosyne/fr/

Distributed under the BSD-2-Clause License

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
csfont = {'fontname':'Times New Roman'}
import mpl_toolkits.axes_grid1 as add_axis
import mpl_toolkits.axisartist 

os.chdir('C://Users//Mnemosyne//Documents//Python Scripts//InverseModelBirdsong//src//Ganguli_Hahnloser_Model')
import GH_model

#parameters
m=3 #number of motor neurons = rows
n=3 #number of auditory neurons = columns
sigma=0.1 #auditory selectivity 
eta=0.01 #learning rate 
sim_number=1 #number of simulations
end_time=10000 #time

#model
PM=GH_model.GH_model(m,n,sigma,eta,sim_number,end_time);

#load the data
mean_err_GH_lin=np.load('Mean distance from target over'+str(sim_number)+' simulations-linear.npy')
mean_err_GH_nl=np.load('Mean distance from target over'+str(sim_number)+' simulations-nonlinear.npy')
std_err_GH_lin=np.load('Std distance from target over'+str(sim_number)+' simulations-linear.npy')
std_err_GH_nl=np.load('Std distance from target over'+str(sim_number)+' simulations-nonlinear.npy')
W_nl=np.load('Synaptic weights nl.npy')
W_lin=np.load('Synaptic weights lin.npy')

#plots
x_time=np.linspace(0,end_time-1,end_time);   

#Figure1: evolution of the mean distance from the target for the linear model
plt.figure()
ax=plt.subplot()
plt.plot(x_time[::10],mean_err_GH_lin[::10])
ax.fill_between(x_time,mean_err_GH_lin[:,0],mean_err_GH_lin[:,0]+std_err_GH_lin[:,0], color='dodgerblue', alpha=.25)
ax.fill_between(x_time,mean_err_GH_lin[:,0],mean_err_GH_lin[:,0]-std_err_GH_lin[:,0], color='dodgerblue', alpha=.25)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.xlabel('Time', **csfont, fontsize=8)
plt.ylabel('Average error over ' + str(sim_number) + ' simulations', **csfont, fontsize=8)
plt.savefig('GHmodel_linear.pdf') 

#Figure2: evolution of the mean distance from the target for the nonlinear model
plt.figure()
ax=plt.subplot()
plt.plot(x_time[::10],mean_err_GH_nl[::10])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.xlabel('Time')
plt.ylabel('Average distance over ' + str(sim_number) + ' simulations')
plt.savefig('GHmodel_nonlinear.pdf')

#Figure3: evolution of the mean distance from the target cfr linear/nonlinear
plt.figure()
ax=plt.subplot() 
plt.plot(x_time[::10],mean_err_GH_lin[::10],label='Linear Model')
plt.plot(x_time[::10],mean_err_GH_nl[::10],label='Nonlinear Model')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.yscale('log')
plt.xlabel('Time (in number of time steps)', **csfont, fontsize=8)
plt.ylabel('Evolution of the distance', **csfont, fontsize=8)
plt.legend(loc='lower left',fontsize=8)
plt.title('Ganguli-Hahnloser Inverse Model',fontsize=15)
plt.savefig('GHmodel.pdf')

#Figure4: evolution of the mean distance from the target cfr linear/nonlinear
#with log scale plot as an insert
fig, ax1=plt.subplots() 
left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax1.plot(x_time,mean_err_GH_lin,color='dodgerblue',label='Linear Model')
ax1.plot(x_time,mean_err_GH_nl,color='coral',label='Nonlinear Model')
ax1.fill_between(x_time,mean_err_GH_lin[:,0],mean_err_GH_lin[:,0]+std_err_GH_lin[:,0], color='dodgerblue', alpha=.25)
ax1.fill_between(x_time,mean_err_GH_lin[:,0],mean_err_GH_lin[:,0]-std_err_GH_lin[:,0], color='dodgerblue', alpha=.25)
ax1.fill_between(x_time,mean_err_GH_nl[:,0],mean_err_GH_nl[:,0]+std_err_GH_nl[:,0], color='coral', alpha=.25)
ax1.fill_between(x_time,mean_err_GH_nl[:,0],mean_err_GH_nl[:,0]-std_err_GH_nl[:,0], color='coral', alpha=.25)
ax1.set_xlabel('Time (in number of time steps)', **csfont, fontsize=8)
ax1.set_ylabel('Evolution of the distance', **csfont, fontsize=8)
ax1.legend(bbox_to_anchor=(.1, 1., 0.8, .102), loc=3,
        ncol=1, mode="expand", borderaxespad=0., fontsize=8)
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax2.plot(x_time,mean_err_GH_lin,color='dodgerblue')
ax2.plot(x_time,mean_err_GH_nl,color='coral')
ax2.set_yscale('log')
plt.savefig('GHmodel_withLOG.pdf')