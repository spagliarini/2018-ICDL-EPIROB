"""

Copyright (c) 2018, Silvia Pagliarini, Xavier Hinaut, Arthur Leblois
https://github.com/spagliarini

Mnemosyne team, Inria, Bordeaux, France
https://team.inria.fr/mnemosyne/fr/

Distributed under the BSD-2-Clause License

PLOT: effect of parameter sigma on distance and convergence time
(Fig. 5-6)

"""

import os
import numpy as np 
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}

#parameters
sim_number=50
sigma=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7] #tuning width

#To have the data all together 
#inside directory where all the data about sigma>0.02 are stored
os.chdir('C://Users//Mnemosyne//Documents//Python Scripts//InverseModelBirdsong//results//IMsimple_model//SigmaComparison//DeltaT400by400') 

conv=np.zeros((sim_number,np.size(sigma)))
final_dist=np.zeros((sim_number,np.size(sigma)))
for i in range (0, sim_number):
    conv[i,1:]=np.load('Convergence time'+str(i)+'simulation.npy')
    for j in range (1,np.size(sigma)):
        final_dist[i,j]=np.load('Error'+str(i)+'simulation.npy')[j-1,int(conv[i,j])-1]
        
#add the data from 0.02 simulations
os.chdir('C://Users//Mnemosyne//Documents//Python Scripts//InverseModelBirdsong//results//IMsimple_model//SigmaComparison//FixTime0.02//TwentyMillionStep50') 
#convergence time
conv[:,0]=20000000 #if fixed conv time
for i in range (0, sim_number): #over 50 simulations
    final_dist[i,0]=np.load('Error'+str(i)+'simulation.npy')[0,-1] #if fixed conv time
    
#change directory back to the directory where all the data are stored
os.chdir('C://Users//Mnemosyne//Documents//Python Scripts//InverseModelBirdsong//results//IMsimple_model//SigmaComparison//UnifyPlot2') 

np.save('FinalDistAll.npy',final_dist)
np.save('ConvTimeAll.npy',conv)

#When data are already all together
os.chdir('C://Users//Mnemosyne//Documents//Python Scripts//InverseModelBirdsong//results//IMsimple_model//SigmaComparison//UnifyPlot2') 

conv=np.load('ConvTimeAll.npy')
final_dist=np.load('FinalDistAll.npy')
plt.figure()
for j in range (0,np.size(sigma)):
    #Histogram of convergence times
    plt.hist(conv[:,j],30,label='sigma=' + str(sigma[j]))
    plt.legend(fontsize=8)
    plt.xscale('log')
    plt.xlabel('Convergence time (in number of time steps)',**csfont, fontsize=8)
    plt.savefig('Convergence time.pdf')

mean_final_dist=np.mean(final_dist,axis=0)
median_final_dist=np.median(final_dist,axis=0)
std_final_dist=np.std(final_dist,axis=0)
mean_conv=np.mean(conv,axis=0)
std_conv=np.std(conv,axis=0)

plt.figure(figsize=(10,8))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx() #twinx add the secondary axis 
ax1.plot(sigma[1:], mean_conv[1::], color='r', label = 'Convergence time (in number of time steps)')  
ax1.fill_between(sigma[1:],mean_conv[1:],mean_conv[1:]-std_conv[1:], color='r', alpha=.25)
ax1.fill_between(sigma[1:],mean_conv[1:],mean_conv[1:]+std_conv[1:], color='r', alpha=.25)
ax1.plot(sigma[0:2],mean_conv[0:2],'r--')
ax1.fill_between(sigma[0:2],mean_conv[0:2],mean_conv[0:2]-std_conv[0:2], color='r', alpha=.25)
ax1.fill_between(sigma[0:2],mean_conv[0:2],mean_conv[0:2]+std_conv[0:2], color='r', alpha=.25)    
ax2.plot(sigma, mean_final_dist, color='b', label = 'Final distance from the target')
ax2.fill_between(sigma,mean_final_dist,mean_final_dist-std_final_dist, color='b', alpha=.25)
ax2.fill_between(sigma,mean_final_dist,mean_final_dist+std_final_dist, color='b', alpha=.25)    
ax1.spines['top'].set_color('none')
ax2.spines['top'].set_color('none')
ax1.set_yscale('log')
ax1.set_xlim(0,0.72)
ax1.set_xticks(sigma)
ax1.set_xticklabels(sigma, rotation=45)
ax1.set_xlabel('Auditory selectivity $\sigma$', **csfont, fontsize=8)
ax1.set_ylabel('Convergence time', **csfont, fontsize=8)
ax2.set_yscale('log')
ax2.set_ylabel('Distance from the target', **csfont, fontsize=8)
fig.legend(bbox_to_anchor=(.1, .95, 0.85, .1), loc=3,
       ncol=2, mode="expand", borderaxespad=0, fontsize=8)
fig.tight_layout() #to avoid the cut of xlabel and right ylabel
plt.savefig('SigmaVsDistance-mean.pdf')

#To compare networks of different dimension
os.chdir('C://Users//Mnemosyne//Documents//Python Scripts//InverseModelBirdsong//results//IMsimple_model//mBYnNeuronModel//ComparisonFix//end1000000')  

neurons_num=[1, 2, 3, 4, 5, 6, 7] #variant dimension (motor or auditory)
neurons_fix=3 #fix dimension (motor or auditory)

conv_M=np.zeros((sim_number,np.size(neurons_num))) #fixed motor dim
final_dist_M=np.zeros((sim_number,np.size(neurons_num)))
conv_A=np.zeros((sim_number,np.size(neurons_num))) #fixed auditory dim
final_dist_A=np.zeros((sim_number,np.size(neurons_num)))
mean_conv=np.zeros((np.size(neurons_num),2)) #mean over M and A fixed
mean_final_dist=np.zeros((np.size(neurons_num),2))
std_final_dist=np.zeros((np.size(neurons_num),2)) #std over M and A fixed
std_conv=np.zeros((np.size(neurons_num),2))
for i in range(0,sim_number):
    for j in range(1,np.size(neurons_num)):
        conv_M[i,j]=np.load('Convergence time' + str(i) +'simulation'+str(neurons_fix)+ ' ' + str(neurons_num[j])+'.npy')
        conv_A[i,j]=np.load('Convergence time' + str(i) +'simulation'+str(neurons_num[j])+ ' ' + str(neurons_fix)+'.npy')
        final_dist_M[i,j]=np.load('Error'+str(i)+'simulation'+str(neurons_fix)+ ' ' + str(neurons_num[j])+'.npy')[0,int(conv_M[i,j])-1]
        final_dist_A[i,j]=np.load('Error'+str(i)+'simulation'+str(neurons_num[j])+ ' ' + str(neurons_fix)+'.npy')[0,int(conv_A[i,j])-1]

mean_conv[:,0]=np.mean(conv_M,axis=0)
mean_conv[:,1]=np.mean(conv_A,axis=0)
std_conv[:,0]=np.std(conv_M,axis=0)
std_conv[:,1]=np.std(conv_A,axis=0)

mean_final_dist[:,0]=np.mean(final_dist_M,axis=0)
mean_final_dist[:,1]=np.mean(final_dist_A,axis=0)
std_final_dist[:,0]=np.std(final_dist_M,axis=0)
std_final_dist[:,1]=np.std(final_dist_A,axis=0)

np.save('Mean_final_dist.npy',mean_final_dist)
np.save('Std_final_dist.npy',std_final_dist) 
np.save('Mean_conv.npy',mean_conv)
np.save('Std_conv.npy',std_conv) 
    
mean_final_dist=np.load('Mean_final_dist.npy')[1::] #start is 1 because we exluded the case m=n=1
std_final_dist=np.load('Std_final_dist.npy')[1::]
mean_conv=np.load('Mean_conv.npy')[1::]
std_conv=np.load('Std_conv.npy')[1::]
neurons_num=neurons_num[1::] #excluding the case m=n=1

fig, ax = plt.subplots()
ax.plot(neurons_num,mean_final_dist[:,0],'cornflowerblue',label='Motor dim=3, Auditory dim=2:7')
ax.fill_between(neurons_num,mean_final_dist[:,0],mean_final_dist[:,0]-std_final_dist[:,0], color='cornflowerblue', alpha=.25)
ax.fill_between(neurons_num,mean_final_dist[:,0],mean_final_dist[:,0]+std_final_dist[:,0], color='cornflowerblue', alpha=.25)
ax.plot(neurons_num,mean_final_dist[:,1],'b',label='Auditory dim=3, Motor dim=2:7')
ax.fill_between(neurons_num,mean_final_dist[:,1],mean_final_dist[:,1]-std_final_dist[:,1], color='b', alpha=.25)
ax.fill_between(neurons_num,mean_final_dist[:,1],mean_final_dist[:,1]+std_final_dist[:,1], color='b', alpha=.25)
ax.set_yscale('log')
ax.set_xticks(neurons_num)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_xlabel('Number of neurons in the network', **csfont, fontsize=8)
ax.set_ylabel('Distance from the target',**csfont, fontsize=8)
ax.set_ylim(0,0.1)
ax.legend(loc='upper left',fontsize=15)
fig.tight_layout() #to avoid the cut of labels
fig.savefig('Distance at convergence time varying network dimension.pdf')    
    
fig, ax = plt.subplots()
ax.plot(neurons_num,mean_conv[:,0],'salmon',label='Motor dim=3, Auditory dim=2:7')
ax.fill_between(neurons_num,mean_conv[:,0],mean_conv[:,0]-std_conv[:,0], color='salmon', alpha=.25)
ax.fill_between(neurons_num,mean_conv[:,0],mean_conv[:,0]+std_conv[:,0], color='salmon', alpha=.25)
ax.plot(neurons_num,mean_conv[:,1],'r',label='Auditory dim=3, Motor dim=2:7')
ax.fill_between(neurons_num,mean_conv[:,1],mean_conv[:,1]-std_conv[:,1], color='r', alpha=.25)
ax.fill_between(neurons_num,mean_conv[:,1],mean_conv[:,1]+std_conv[:,1], color='r', alpha=.25)
ax.set_yscale('log')
ax.set_xticks(neurons_num)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.set_yticks(ax.get_yticks() + 10000)
ax.set_ylim(10000, 3000000)
ax.set_xlabel('Number of neurons in the network', **csfont, fontsize=8)
ax.set_ylabel('Convergence time (in number of time steps)', **csfont, fontsize=8)
ax.legend(loc='lower right', fontsize=15)
fig.tight_layout() #to avoid the cut of labels
fig.savefig('Distance time varying network dimension.pdf')