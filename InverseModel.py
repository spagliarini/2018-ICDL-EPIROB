"""
INVERSE MODEL

Copyright (c) 2018, Silvia Pagliarini, Xavier Hinaut, Arthur Leblois
https://github.com/spagliarini

Mnemosyne team, Inria, Bordeaux, France
https://team.inria.fr/mnemosyne/fr/

Distributed under the BSD-2-Clause License

"""
import numpy as np
import matplotlib.pyplot as plt
import time 

##Learning rule 
#Classic Hebbian learning rule
def IM_simple_classic(eta,M,A,W): 
    """
    ....

    Inputs:
        - eta: learning rate
        - M: motor exploration 
        - A: auditory activity
        - W: current weights matrix
    """
    DeltaW=eta*M*A
    W=W+DeltaW

    return W

#Normalized Hebbian learning rule: weights' mean equals to 1
def IM_simple_2avrg(m,n,eta,M,A,W,option_rc): 
    """
    Normalisation of W with respect to its mean (equal to 1)

    Inputs:
        - eta,M,A,W : similar to IM_simple_classic() method
        - m: number of motor neurons
        - n: number of auditive neurons 
        - option_rc: to choose if we normalized over lines or columns
    """
    if (option_rc==0):
        W_avrg=np.sum(W,axis=0)/m
        W=W/W_avrg
    elif (option_rc==1):
        W_avrg=np.sum(W,axis=1)/n
        W=(W.T/W_avrg).T

    return IM_simple_classic(eta=eta,M=M,A=A,W=W)

#Normalized Hebbian learning rule: weights' norm equals to 1
def IM_simple_2norm(m,n,eta,M,A,W,option_rc): 
    """
    Normalization of W with respect to its norm  (equal to 1)
    
    Inputs:
        - eta,M,A,W : similar to IM_simple_classic() method
        - m,n,option_rc: similar to IM_simple_2avrg() method
    """
    if (option_rc==0):
        W_norm=np.linalg.norm(W,axis=0)
        W=W/W_norm

    elif (option_rc==1):
        W_norm=np.linalg.norm(W,axis=1)
        W=(W.T/W_norm).T

    return IM_simple_classic(eta=eta,M=M,A=A,W=W)

#Normalized Hebbian learning rule: weights' mean maximum equals to 1
def IM_simple_3avrg(m,n,eta,M,A,W,option_rc): 
    """
    Normalization of W with respect to its mean (as maximum 1)
    
    Inputs:
        - eta,M,A,W : similar to IM_simple_classic() method
        - m,n,option_rc: similar to IM_simple_2avrg() method
    """
    if (option_rc==0):
        W_avrg=np.sum(W,axis=0)/m
        for j in range(0,n):
            if (W_avrg[j]>=1):
                W[:,j]=W[:,j]/W_avrg[j]

    elif (option_rc==1):
        W_avrg=np.sum(W,axis=1)/n
        for j in range(0,m):
            if (W_avrg[j]>=1):
                W[j,:]=W[j,:]/W_avrg[j]

    return IM_simple_classic(eta=eta,M=M,A=A,W=W)

#Normalized Hebbian learning rule: weights' norm maximum equals to 1
def IM_simple_3norm(m,n,eta,M,A,W,option_rc): 
    """
    Normalization of W with respect to its norm(as maximum 1)
    
    Inputs:
        - eta,M,A,W : similar to IM_simple_classic() method
        - m,n,option_rc: similar to IM_simple_2avrg() method
    """
    if (option_rc==0):
        W_norm=np.linalg.norm(W,axis=0)
        for j in range(0,n):
            if (W_norm[j]>=1):
                W[:,j]=W[:,j]/W_norm[j]

    elif (option_rc==1):
        W_norm=np.linalg.norm(W,axis=1)
        for j in range(0,m):
            if (W_norm[j]>=1):
                W[j,:]=W[j,:]/W_norm[j]

    return IM_simple_classic(eta=eta,M=M,A=A,W=W)

#Normalized Hebbian learning rule: variation of the weights rescaled by a 
#                                  decreasing factor depending on weights' mean
def IM_simple_4avrg(m,n,eta,M,A,W,option_rc): 
    """
    DeltaW rescaled by a decreasing factor depending on the mean of W
    
    Inputs:
        - eta,M,A,W : similar to IM_simple_classic() method
        - m,n,option_rc: similar to IM_simple_2avrg() method
    """
    if (option_rc==0):
        W_avrg=np.sum(W,axis=0)/m
        DeltaW=eta*M*A*(1-W_avrg)
        W=W+DeltaW

        return W, DeltaW

    if (option_rc==1):
        W_avrg=np.sum(W,axis=1)/n
        DeltaW=((eta*M*A).T*(1-W_avrg)).T
        W=W+DeltaW

        return W, DeltaW

#Normalized Hebbian learning rule: variation of the weights rescaled by a 
#                                  decreasing factor depending on weights' norm
def IM_simple_4norm(m,n,eta,M,A,W,option_rc): 
    """
    DeltaW rescaled by a decreasing factor depending on the norm of W 
    
    Inputs:
        - eta,M,A,W : similar to IM_simple_classic() method
        - m,n,option_rc: similar to IM_simple_2avrg() method
    """
    if (option_rc==0):
        W_norm=np.linalg.norm(W,axis=0)
        DeltaW=eta*M*A*(1-W_norm)
        W=W+DeltaW

        return W

    if (option_rc==1):
        W_norm=np.linalg.norm(W,axis=1)
        DeltaW=((eta*M*A).T*(1-W_norm)).T
        W=W+DeltaW

        return W
    
##Time decoder
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(te-ts)
        return result
    return timed

##Simple inverse model VS all normalizations
@timeit
def IM_simple_normalized(a,b,m,n,seed,expl,sigma,eta,sim_number,end_time,option_rc):
    """
        Inputs:
        - eta: similar to IM_simple_classic() method
        - m,n,option_rc: similar to IM_simple_2avrg() method
        - a,b : interval wich define the target space
        - seed: maximum value fot the weight at he beginning
        - expl: maximum value for the motor exploration
    
        Output: simple inverse model  + normalization
    
        Variables in code:
        - PM: motor target
        - M: motor exploratio
        - A: auditory activity
        - W: synaptic weights
        - DeltaW: synaptic weights variation    
    """
    ##Inizialization of variables
    err_wn=np.zeros((sim_number,end_time)) #inizialization of error (one for each type of normalization)
    err_2avrg=np.zeros((sim_number,end_time))
    err_2norm=np.zeros((sim_number,end_time))
    err_3avrg=np.zeros((sim_number,end_time))
    err_3norm=np.zeros((sim_number,end_time))
    err_4avrg=np.zeros((sim_number,end_time))
    err_4norm=np.zeros((sim_number,end_time))
    #
    counter=0 #simulation counter
    while (counter<sim_number):
        #ideal motor pattern definition
        PM=(np.random.uniform(a,b,(m,n))) #initializing motor patterns at t=0 for each simulation
        PM_mean=PM/np.mean(PM,axis=0)
        PM_norm=PM/np.linalg.norm(PM,axis=0)
        #inizializing the weight matrix, and eventually the variation of weights, at t=0 for each simulation, for each type of normalization:
        W_wn=(np.random.uniform(0,seed,(m,n)))    #classic model without any normalization
        W_2avrg=W_wn #normalization putting the average of each column equal to 1
        W_2norm=W_wn #normalization putting the norm of each column equal to 1
        W_3avrg=W_wn #normalization putting the average of each column at maximum 1
        W_3norm=W_wn #normalization putting the norm of each column at maximum 1
        W_4avrg=W_wn #normalization scaling the increment in W using a factor dependind on the average of each column
        W_4norm=W_wn #normalization scaling the increment in W using a decreasing factor depending on the norm of each column
        #
        for t in range(0,end_time):
            M=np.random.uniform(0,expl,(m,1)) #inizializing the motor activity at each time
            A_mean=np.exp((-(np.transpose(np.linalg.norm(PM_mean-M,axis=0)))**2)/(2*m*sigma**2)) #auditory activity
            A_norm=np.exp((-(np.transpose(np.linalg.norm(PM_norm-M,axis=0)))**2)/(2*m*sigma**2)) #auditory activity
            ##Weight's matrixes
            W_wn=IM_simple_classic(eta,M,A_mean,W_wn) # without normalization
            W_2avrg=IM_simple_2avrg(m,n,eta,M,A_mean,W_2avrg,option_rc) #2 average
            W_2norm=IM_simple_2norm(m,n,eta,M,A_norm,W_2norm,option_rc) #2 norm
            W_3avrg=IM_simple_3avrg(m,n,eta,M,A_mean,W_3avrg,option_rc) #3 average
            W_3norm=IM_simple_3norm(m,n,eta,M,A_norm,W_3norm,option_rc) #3 norm
            W_4avrg=IM_simple_4avrg(m,n,eta,M,A_mean,W_4avrg,option_rc)[0] #4 average
            W_4norm=IM_simple_4norm(m,n,eta,M,A_norm,W_4norm,option_rc) #4 norm
                      
            ##Error
            if (option_rc==0):
                err_wn[counter,t]=np.linalg.norm(PM-W_wn)/m       #without normalization
                err_2avrg[counter,t]=np.linalg.norm(PM_mean-W_2avrg)/m #2 average
                err_2norm[counter,t]=np.linalg.norm(PM_norm-W_2norm)/m #2 norm
                err_3avrg[counter,t]=np.linalg.norm(PM_mean-W_3avrg)/m #3 average
                err_3norm[counter,t]=np.linalg.norm(PM_norm-W_3norm)/m #3 norm
                err_4avrg[counter,t]=np.linalg.norm(PM_mean-W_4avrg)/m #4 average
                err_4norm[counter,t]=np.linalg.norm(PM_norm-W_4norm)/m #4 norm

            if (option_rc==1):
                err_wn[counter,t]=np.linalg.norm(PM-W_wn)/n       #without normalization
                err_2avrg[counter,t]=np.linalg.norm(PM_mean-W_2avrg)/n #2 average
                err_2norm[counter,t]=np.linalg.norm(PM_norm-W_2norm)/n #2 norm
                err_3avrg[counter,t]=np.linalg.norm(PM_mean-W_3avrg)/n #3 average
                err_3norm[counter,t]=np.linalg.norm(PM_norm-W_3norm)/n #3 norm
                err_4avrg[counter,t]=np.linalg.norm(PM_mean-W_4avrg)/n #4 average
                err_4norm[counter,t]=np.linalg.norm(PM_norm-W_4norm)/n #4 norm

        
        counter=counter+1 #pass to next simulation
        
    np.save('Distance_simple',err_wn)
    np.save('Distance_2avrg',err_2avrg)
    np.save('Distance_2norm',err_2norm)
    np.save('Distance_3avrg',err_3avrg)
    np.save('Distance_3norm',err_3norm)    
    np.save('Distance_4avrg',err_4avrg)
    np.save('Distance_4norm',err_4norm)    
    return err_wn, err_2avrg,err_2norm, err_3avrg, err_3norm, err_4avrg, err_4norm

##Simple inverse model VS the normalization we chose to use
@timeit
def IM_simple(a,b,m,n,seed,expl,sigma,eta,sim_number,end_time,option_rc):
    """
        Inputs:
        - eta: similar to IM_simple_classic() method
        - m,n,option_rc: similar to IM_simple_2avrg() method
        - a,b : interval wich define the target space
        - seed: maximum value fot the weight at he beginning
        - expl: maximum value for the motor exploration
    
        Output: normalization using DeltaW multiplied by a decreasing factor
                no threshold in the distance 
                only one value of sigma
                
        Variables in code:
        - PM: motor target
        - M: motor exploratio
        - A: auditory activity
        - W: synaptic weights
        - DeltaW: synaptic weights variation
    """
    #inizialization of error
    err_wn=np.zeros((sim_number,end_time))
    err_4avrg=np.zeros((sim_number,end_time))
    err_4norm=np.zeros((sim_number,end_time))
    #to save weights at each time step
    W=np.zeros((m*n, end_time))
    #simulation counter
    counter=0
    while (counter<sim_number):
        #ideal motor pattern definition
        PM=(np.random.uniform(a,b,(m,n))) #initializing motor patterns at t=0 for each simulation
        PM_mean=PM/np.mean(PM,axis=0)
        PM_norm=PM/np.linalg.norm(PM,axis=0)
        np.save('Ideal motor pattern' +str(counter) +'simulation.npy', PM)
        #inizializing the weight matrix, and eventually the variation of weights, at t=0 for each simulation, for each type of normalization:
        W_wn=(np.random.uniform(0,seed,(m,n)))    #classic model without any normalization
        W_4avrg=W_wn #normalization putting the average of each column equal to 1
        W_4norm=W_wn
        for t in range(0,end_time):             #
            M=np.random.uniform(0,expl,(m,1)) #inizializing the motor activity at each time
            A_mean=np.exp((-(np.transpose(np.linalg.norm(PM_mean-M,axis=0)))**2)/(2*m*sigma**2)) #auditory activity
            A_norm=np.exp((-(np.transpose(np.linalg.norm(PM_norm-M,axis=0)))**2)/(2*m*sigma**2)) #auditory activity
            ##Weight's matrixes
            W_wn=IM_simple_classic(eta,M,A_mean,W_wn) # without normalization
            W_4avrg=IM_simple_4avrg(m,n,eta,M,A_mean,W_4avrg,option_rc)[0] #4 average
            W_4norm=IM_simple_4norm(m,n,eta,M,A_norm,W_4norm,option_rc)[0] #4 norm

            #Save the weights to plot the evolution in time
            if counter==sim_number-1:
                W[:,t]=np.reshape(W_4avrg,(m*n,))
            ##Error
            if (option_rc==0):
                err_wn[counter,t]=np.linalg.norm(PM-W_wn)/m       #without normalization
                err_4avrg[counter,t]=np.linalg.norm(PM_mean-W_4avrg)/m #4 average
                err_4norm[counter,t]=np.linalg.norm(PM_norm-W_4norm)/m #4 norm
            if (option_rc==1):
                err_wn[counter,t]=np.linalg.norm(PM-W_wn)/n       #without normalization
                err_4avrg[counter,t]=np.linalg.norm(PM_mean-W_4avrg)/n #4 average
                err_4norm[counter,t]=np.linalg.norm(PM_norm-W_4norm)/n #4 norm

        np.save('Evolution of weights for'+ str(counter) +'simulation.npy',W)
        np.save('Weights matrix' + str(counter) + 'simulation.npy',W_4avrg)
        np.save('Error'+ str(counter) +'simulation.npy',err_4avrg)
        counter=counter+1

    return err_wn, err_4avrg, err_4norm
    pass

##To compare sigma using a threshold computed starting from the distance
@timeit
def IM_sigma_threshold(a,b,m,n,seed,expl,sigma,eta,sim_number,end_time,option_rc):
    """
        Inputs:
        - eta: similar to IM_simple_classic() method
        - m,n,option_rc: similar to IM_simple_2avrg() method
        - a,b : interval wich define the target space
        - seed: maximum value fot the weight at he beginning
        - expl: maximum value for the motor exploration
        - end_time: eventual time to exit the simulation even if the threshold has not been reached
    
        Output: normalization using DeltaW multiplied by a decreasing factor
                threshold in the distance 
                different values of sigma can be tested
       
        Variables in code:
        - PM: motor target
        - M: motor exploratio
        - A: auditory activity
        - W: synaptic weights
        - DeltaW: synaptic weights variation
    """

    ##Inizialization of variables
    dist=np.zeros((sim_number,np.size(sigma),end_time))  #inizialization of distance for simple model with normalization
    #
    points=end_time/10  #
    counter=0 #simulation counter
    tol=10e-9 #error threshold
    while (counter<sim_number): #loop over the number of simulations
        PM=(np.random.uniform(a,b,(m,n))) #initializing motor patterns at t=0 for each simulation
        PM=PM/np.mean(PM,axis=0) #applying normalization also to the ideal motor pattern
        np.save('Ideal motor pattern' +str(counter) +'simulation '+str(m)+ ' ' + str(n)+'.npy', PM)
        #inizializing the weight matrix and the variation matrix
        W_seed=(np.random.uniform(0,seed,1,(m,n)))  #classic model without any normalization
        W=np.zeros((np.size(sigma),m,n)) #np.size(sigma) matrixes of dim mxn
        W[:]=W_seed
        DeltaW=np.zeros((np.size(sigma),m,n))
        M=np.zeros((end_time,m,1))
        #to save random motor activity and step distance
        dist_step=np.zeros((np.size(sigma),end_time,n))
        #threshold update inizalization
        eps=np.ones((np.size(sigma),))  #np.size(sigma)x1 vector
        #time step counter
        t=0
        conv_time=np.zeros(np.size(sigma),)
        while (t<end_time) and (eps[0]>tol): #or (eps[1]>tol) or (eps[2]>tol) or (eps[3]>tol) or (eps[4]>tol) or (eps[5]>tol):
        #in any case we fixed a maximum number of time steps so we exit the simulation if end_time is reached
            #inizializing the motor activity at each time
            M[t,:,:]=np.random.uniform(0,expl,(m,1))
            #auditory activity
            A=np.ones((np.size(sigma),n,end_time)) #matrix n x np.size(sigma)x1 , each column correspond to the auditory activity of a value of sigma
            for i in range(0,np.size(sigma)):
                A[i,0:3,t]=np.exp((-(np.transpose(np.linalg.norm(PM[:,0:3]-M[t,:,:],axis=0)))**2)/(2*m*sigma[i]**2))
                #Weight's matrixes
                if (eps[i]>tol):
                    W[i,:], DeltaW[i,:]=IM_simple_4avrg(m,n,eta,M[t,:,:],A[i,:,t],W[i,:],option_rc) #4 average
                    conv_time[i]=conv_time[i]+1

                ##Distance
                dist[counter,i,t]=np.linalg.norm(PM-W[i,:])/m
                ##Update epsilon each 400 time steps
                if (t>400)==True:
                    if (t%400==0):
                        eps[i]=np.mean(dist[counter,i,t-800:t-400])-np.mean(dist[counter,i,t-400:t])

            t=t+1

        ##Save data
        ##Save data
        np.save('Error' +str(counter) +'simulation.npy', dist[counter,:,:])
        np.save('Convergence time' + str(counter) + 'simulation.npy',conv_time)
        
        counter=counter+1 #to next simulation

    return 

##To compare sigma using a fixed end_time to exit simulations
@timeit
def IM_sigma(a,b,m,n,seed,expl,sigma,eta,sim_number,end_time,option_rc):
    """
        Inputs:
        - eta: similar to IM_simple_classic() method
        - m,n,option_rc: similar to IM_simple_2avrg() method
        - a,b : interval wich define the target space
        - seed: maximum value fot the weight at he beginning
        - expl: maximum value for the motor exploration
        - end_time: eventual time to exit the simulation even if the threshold has not been reached
    
        Output: normalization using DeltaW multiplied by a decreasing factor
                threshold in the distance 
                different values of sigma can be tested
                
        Variables in code:
        - PM: motor target
        - M: motor exploratio
        - A: auditory activity
        - W: synaptic weights
        - DeltaW: synaptic weights variation
    """

    ##Inizialization of variables
    dist=np.zeros((sim_number,np.size(sigma),end_time,n))  #inizialization of distance for simple model with normalization

    #
    points=end_time/10
    counter=0 #simulation counter
    tol=10e-9 #error threshold
    eps=np.ones((np.size(sigma),))
    for counter in range(sim_number): #loop over the number of simulations
        PM=(np.random.uniform(a,b,(m,n))) #initializing motor patterns at t=0 for each simulation
        #TODO: put the 0.5 and 1.5 as parameters
        PM=PM/np.mean(PM,axis=0) #applying normalization also to the ideal motor pattern        
        np.save('Ideal motor pattern' +str(counter) +'simulation.npy', PM)
        #inizializing the weight matrix and the variation matrix
        W_seed=(np.random.uniform(0,seed,(m,n)))  #classic model without any normalization

        W=np.zeros((np.size(sigma),m,n)) #np.size(sigma) matrixes of dim mxn
        W[:]=W_seed 
        
        DeltaW=np.zeros((np.size(sigma),m,n))
        M=np.zeros((end_time,m,1))
        #to save convergence time
        conv_time=np.zeros(np.size(sigma),)
        W_save=np.zeros((np.size(sigma),m*n,end_time))
        A=np.ones((np.size(sigma),n,end_time)) #matrix n x np.size(sigma)x1 , each column correspond to the auditory activity of a value of sigma
        for t in range(0,end_time):
            #inizializing the motor activity at each time
            M[t,:,:]=np.random.uniform(0,expl,(m,1))
            #auditory activity
            for i in range(0,np.size(sigma)):
                A[i,0:3,t]=np.exp((-(np.transpose(np.linalg.norm(PM[:,0:3]-M[t,:,:],axis=0)))**2)/(2*m*sigma[i]**2))
                #Weight's matrixes
                W[i,:], DeltaW[i,:]=IM_simple_4avrg(m,n,eta,M[t,:,:],A[i,:,t],W[i,:],option_rc) #4 average
                W_save[i,:,t]=np.reshape(W[i,:],(m*n,))
                 
                ##Distance
                dist[counter,i,t,:]=np.linalg.norm(PM-W[i,:],axis=0)
                ##Update epsilon each 400 time steps
                if (t>400)==True:
                    if (t%400==0):
                        eps[i]=np.mean(dist_tot[counter,i,t-800:t-400])-np.mean(dist_tot[counter,i,t-400:t])
                        if eps[i]<tol:
                            conv_time[i]=t
               
        ##Save data
        np.save('Error' +str(counter) +'simulation.npy', dist[counter,:,:,:])
        np.save('Convergence time' + str(counter) + 'simulation.npy',conv_time)

        counter=counter+1 #pass to next simulation
