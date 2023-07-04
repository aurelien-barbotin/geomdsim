#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:29:20 2023

@author: aurelienb
"""
from pyimfcs.class_imFCS import StackFCS
from pyimfcs.fitting import Fitter, make_fitp0
from tifffile import imsave
import os
import pandas as pd
def process_stack(savepath,tirf_simulator=None, nsums=[2,3],
                           trajectory=None,
                           fitter = None, 
                           chi_threshold = 0.03, ith=0.8,initial_guess_D=1, 
                           delete_tif = False, shifts = None):
    """Path is where to create the stack"""
    raw_stack = tirf_simulator.get_stack()
    if shifts is not None:
        sx,sy=shifts
        raw_stack = raw_stack[:,sx:,sy:]
    path=savepath+"stack.tif"
    
    imsave(path, raw_stack)
    
    stack = StackFCS(path, background_correction = True,           
                         clipval = 0)
    default_dt = trajectory.dt
    stack.dt = default_dt
    stack.xscale = tirf_simulator.psize
    stack.yscale = tirf_simulator.psize
    
    make_fitp0("2D",[lambda x:max(10**-3,1/x[0,1]), lambda x: initial_guess_D])
    for nSum in nsums:
        stack.correlate_stack(nSum)
    if fitter is None:
        sigmaxy = tirf_simulator.sigma_psf
        parameters_dict = {"a":stack.yscale, "sigma":sigmaxy,"mtype":"2D","ginf":True}
        ft = Fitter(parameters_dict)
    else:
        ft = fitter
    
    stack.fit_curves(ft,xmax=None)
    
    stack.save(exclude_list=['intensity_traces'])
    if delete_tif:
        os.remove(path)
    pars = trajectory.parameters_dict()
    pars.update(tirf_simulator.parameters_dict())
    pars = pd.DataFrame(pars, index=[0])
    pars.to_excel(savepath+"parameters.xlsx")