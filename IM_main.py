# -*- coding: utf-8 -*-
"""
INVERSE MODEL BIRDSONG

Copyright (c) 2018, Silvia Pagliarini, Xavier Hinaut, Arthur Leblois
https://github.com/spagliarini

Mnemosyne team, Inria, Bordeaux, France
https://team.inria.fr/mnemosyne/fr/

Distributed under the BSD-2-Clause License
"""

import os
import numpy as np

import InverseModel

def simple_vs_all(a, b, m, n, seed, expl, sigma, eta, sim_number, end_time, option_rc):
    InverseModel.IM_simple_normalized(a,b,m,n,seed,expl,sigma,eta,sim_number,end_time,option_rc)

def simple_vs_one(a, b, m, n, seed, expl, sigma, eta, sim_number, end_time, option_rc):
    InverseModel.IM_simple(a,b,m,n,seed,expl,sigma,eta,sim_number,end_time,option_rc)

def sigma_end_time(a, b, m, n, seed, expl, sigma, eta, sim_number, end_time, option_rc):
    InverseModel.IM_sigma(a,b,m,n,seed,expl,sigma,eta,sim_number,end_time,option_rc)

def sigma_threshold(a, b, m, n, seed, expl, sigma, eta, sim_number, end_time, option_rc):
    InverseModel.IM_sigma_threshold(a,b,m,n,seed,expl,sigma,eta,sim_number,end_time,option_rc)
    
if __name__=="__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parameters", type=str, help="json file containing parameters for experiment (root directory is assumed to be params/)")
    parser.add_argument("-t", "--type_sim", type=str, help="which simulation type should be chosen")
    args = parser.parse_args()

    if args.parameters is None:
        raise Exception("No json parameter file provided. Please run the file in this way:\npython IM_main.py -p params/default_param.json")
    with open(args.parameters, 'r') as f:
        params = json.load(f)
    a = params['a'] #left boundary of the target space PM
    b = params['b'] #right boundary of the target space PM
    m = params['m'] #number of motor neurons = rows
    n = params['n'] #number of auditory neurons = columns
    seed = params['seed'] #right boundary of the synaptic weights = max value of the weights at the beginning
    expl = params['expl'] #right boundary of the motor exploration = max value of the motor exploration at the each time step
    sigma = params['sigma'] #auditory selectivity tested values=0.02,0.05,0.1,0.2,0.3,0.5,0.7
    eta = params['eta'] #learning rate
    sim_number = params['sim_number'] #number of simulations
    end_time = params['end_time'] #end time
    option_rc = params['option_rc'] #to choose if we are normalizing over right or columns 
    print("retrieve all params")

    if args.type_sim == 'simple_vs_all':
        simple_vs_all(a=a, b=b, m=m, n=n, seed=seed, expl=expl, sigma=sigma, eta=eta, sim_number=sim_number, end_time=end_time, option_rc=option_rc)
    elif args.type_sim == 'simple_vs_one':
        simple_vs_one(a=a, b=b, m=m, n=n, seed=seed, expl=expl, sigma=sigma, eta=eta, sim_number=sim_number, end_time=end_time, option_rc=option_rc)
    elif args.type_sim == 'sigma_threshold':
        sigma_threshold(a=a, b=b, m=m, n=n, seed=seed, expl=expl, sigma=sigma, eta=eta, sim_number=sim_number, end_time=end_time, option_rc=option_rc)
    elif args.type_sim == 'sigma_end_time':
        sigma_end_time(a=a, b=b, m=m, n=n, seed=seed, expl=expl, sigma=sigma, eta=eta, sim_number=sim_number, end_time=end_time, option_rc=option_rc)
