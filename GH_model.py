"""

HAHNLOSER & GANGULI MODEL 
with a nonlinear sensory response 

Copyright (c) 2018, Silvia Pagliarini, Xavier Hinaut, Arthur Leblois
https://github.com/spagliarini

Mnemosyne team, Inria, Bordeaux, France
https://team.inria.fr/mnemosyne/fr/

Distributed under the BSD-2-Clause License
"""
import numpy as np 

def GH_model(m,n,sigma,eta,sim_number,end_time):
    
    #error definition 
    err_GH_lin=np.zeros((sim_number,end_time))
    err_GH_nl=np.zeros((sim_number,end_time))
    
    #simulation counter
    counter=0
    while(counter<sim_number):
        #target motor pattern
        PM=np.random.uniform(0,2,(m,n)) 
        #check that condition number of matrix PM to have a well-conditioned 
        #linear system
        while np.linalg.cond(PM)>10:
            PM=np.random.uniform(0,2,(m,n))
        #due to normalization we need to normalize also the target
        PM=PM/np.mean(PM,axis=0)
        np.save('Ideal motor pattern simulation'+str(counter)+'.npy',PM)
        
        #weight's matrix
        W_GH_lin=np.random.uniform(0,0.01,(m,n))  
        W_GH_nl=np.random.uniform(0,0.01,(m,n))  
        
        #loop over the time
        for t in range(0,end_time):
            #motor activity
            M=np.random.uniform(0,2,(m,1)) 
            #linear auditory activity
            A=np.linalg.solve(PM,M)
            #nonlinear auditory activity
            Anl=np.exp((-((np.linalg.norm(PM-M,axis=0)))**2)/(2*m*sigma**2))
            #learning rule 
            DeltaW_GH_lin=eta*np.dot((M-np.dot(W_GH_lin,A)),np.transpose(A))
            DeltaW_GH_nl=eta*np.dot((M-np.dot(W_GH_lin,Anl)),np.transpose(Anl))
            #update the weights
            W_GH_lin=W_GH_lin+DeltaW_GH_lin
            W_GH_nl=W_GH_nl+DeltaW_GH_nl
            
            #distance from the current output and the target 
            err_GH_lin[counter,t]=np.linalg.norm(PM-W_GH_lin)/m
            err_GH_nl[counter,t]=np.linalg.norm(PM-W_GH_nl)/m
            
        counter=counter+1
    
    #mean distance between the simulation output and the target    
    mean_err_GH_lin=np.zeros((end_time,1))
    std_err_GH_lin=np.zeros((end_time,1))
    mean_err_GH_nl=np.zeros((end_time,1))
    std_err_GH_nl=np.zeros((end_time,1))
    for k in range(0,end_time):
        mean_err_GH_lin[k]=np.mean(err_GH_lin[:,k])
        std_err_GH_lin[k]=np.std(err_GH_lin[:,k])
        mean_err_GH_nl[k]=np.mean(err_GH_nl[:,k])
        std_err_GH_nl[k]=np.std(err_GH_nl[:,k])
        
    np.save('Distance from target over'+str(sim_number)+' simulations-linear.npy',err_GH_lin)
    np.save('Distance from target over'+str(sim_number)+' simulations-nonlinear.npy',err_GH_nl)
    np.save('Mean distance from target over'+str(sim_number)+' simulations-linear.npy',mean_err_GH_lin)
    np.save('Mean distance from target over'+str(sim_number)+' simulations-nonlinear.npy',mean_err_GH_nl)    
    np.save('Std distance from target over'+str(sim_number)+' simulations-linear.npy',std_err_GH_lin)
    np.save('Std distance from target over'+str(sim_number)+' simulations-nonlinear.npy',std_err_GH_nl)    
    np.save('Synaptic weights nl.npy',W_GH_nl)
    np.save('Synaptic weights lin.npy',W_GH_lin)

    return 
