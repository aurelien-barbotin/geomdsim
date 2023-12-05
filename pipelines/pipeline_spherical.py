#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:40:34 2023

@author: aurelienb
"""

import numpy as np
from geomdsim.trajectory import SphericalTrajectory
from geomdsim.imaging import TIRF_Simulator
from geomdsim.methods import prepare_savefolder
from geomdsim.processing import process_stack

from pyimfcs.export import merge_fcs_results

import time
import matplotlib.pyplot as plt

plt.close('all')
dt = 10**-3 # s
D = 1 # um2/s
nsteps = 50000
nparts = 20
# R = 10 # um

npix_img = 50
psize=0.08
sigma_psf = 0.19
dz_tirf = 0.1
brightness = 20*10**3
z_cutoff_factor = 4
nsums = [2,3,4]
R = 0.35
t0 = time.time()
sigma_psf=0.19

savepath="/home/aurelienb/Data/simulations/2023_09_11_listeria_spherical_J148r035/"
# nparts = max(10,int(1000*(R/10)**2)//2)
for sx in [0,2,4]:
    for sy in [1,3,5]:
        trajectory = SphericalTrajectory(dt,D,nsteps,nparts,R)
        
        imager = TIRF_Simulator(npix_img, npix_img, psize, 
                     sigma_psf, dz_tirf, brightness, dt,
                     z_cutoff_factor)
        
        for j in range(nsteps):
            coords_xyz = trajectory.get_positions()
            imager.generate_frame(coords_xyz)
            trajectory.next_step()
            if j%500==0:
                print("Processing frame {}".format(j))
        
        # trajectory.plot_positions()
        savefolder = prepare_savefolder(savepath)
        
        process_stack(savefolder,tirf_simulator=imager, nsums=nsums,
                                   trajectory=trajectory,
                                   fitter = None, 
                                   chi_threshold = 0.03, ith=0.8,initial_guess_D=1, 
                                   delete_tif = True, shifts=(sx,sy))
        thr = 0.03
        intensity_threshold=0.8
        merge_fcs_results( savefolder+"FCS_results",[savefolder+"stack"+".h5"], 
              ith = intensity_threshold, chi_threshold = thr)

t1 = time.time()
print("Elapsed time : {:.1f} s".format(t1-t0))