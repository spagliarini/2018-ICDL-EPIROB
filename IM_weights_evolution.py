# -*- coding: utf-8 -*-
"""

Copyright (c) 2018, Silvia Pagliarini, Xavier Hinaut, Arthur Leblois
https://github.com/spagliarini

Mnemosyne team, Inria, Bordeaux, France
https://team.inria.fr/mnemosyne/fr/

Distributed under the BSD-2-Clause License
PLOT: Evolution of weights relative to one syllable, so 1 auditive neuron and 3 motor neurons
(Fig. 3)

"""

import os
import numpy as np 
import matplotlib.pyplot as pltimport os
import numpy as np 
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}

os.chdir('C://Users//Mnemosyne//Documents//Python Scripts//InverseModelBirdsong//results//IMsimple_model//mBYnNeuronModel//ExSim') 

##Time evolution of weights
W4avrg=np.load('Weights matrix0simulation_allTime.npy')
PM=np.load('Ideal motor pattern0simulation.npy')

end_time=W4avrg.shape[2]
x_time=np.linspace(0,end_time-1,end_time)

ax=plt.figure()
plt.plot(x_time,W4avrg[0,0,:],'b',x_time,np.ones((end_time,1))*PM[0,0],'b--',x_time,W4avrg[0,1,:],'b',x_time,np.ones((end_time,1))*PM[1,0],'b--',x_time,W4avrg[0,2,:],'b',x_time,np.ones((end_time,1))*PM[2,0],'b--')
plt.ylabel('Evolution of weights',**csfont, fontsize=8)
plt.xlabel('Time (in number of time steps)', **csfont, fontsize=8)
plt.savefig('Evolution of weights.pdf')
