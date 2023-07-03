#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:07:17 2023

@author: aurelienb

objects uused to described trajectories in different geometrical configurations
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
import datetime
import os
from numpy.linalg import norm

from pyimfcs.class_imFCS import StackFCS
from pyimfcs.fitting import Fitter
from pyimfcs.export import merge_fcs_results

import tifffile
import pandas as pd

from geomdsim.methods import spherical2cart, set_axes_equal

# 1/ trajectory simulation: 1/ generate initial conditions 2/ moves these conditions
# 2/ from trajectory to image
# 2/ process the resulting stack, generate folder etc

class Trajectory(object):
    def __init__(self, dt,D,nsteps,nparts, save_coordinates=False,*args, **kwargs):
        """Sets up the trajectories, in physical units. Units used: seconds, 
        micrometers."""
        super().__init__() # is this even useful?
        
        # these four are always there and therefore initialised in the abstract class
        self.dt = dt
        self.D = D
        self.nsteps = nsteps
        self.nparts = nparts
        self.save_coordinates = save_coordinates
        # insert code here for initial conditions
        self.xyz = (0,0,0)
        self.current_frame = 0
        
    def next_step(self):
        """Makes one step: updates values of all coordinates accordingly"""
        self.current_frame+=1
        x,y,z = self.xyz
        return x,y,z
    def get_positions(self):
        """Method to get the positions recentered for imaging. Recentring depends
        on the simulaiton method, e.g in a sphere you want to add the radius to the z coordinates
        so that we simulate imaging bottom of the sphere"""
        x,y,z=self.xyz[:,0], self.xyz[:,1], self.xyz[:,2]
        return x,y,z
    def export_params(self):
        pass

    def plot_positions(self):
        plt.figure()
        ax = plt.axes(projection='3d')
        # set_axes_equal(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        x,y,z = self.get_positions()
        ax.scatter3D(x, y, z,color="C0")
        set_axes_equal(ax)
    def plot_trajectories(self):
        pass
    def parameters_dict(self):
        par_dict= {"dt":self.dt,
                   "D":self.D,
                   "nsteps":self.nsteps,
                   "nparts":self.nparts}
        return par_dict

class SphericalTrajectory(Trajectory):
    def __init__(self, dt,D,nsteps,nparts,R):
        super().__init__(dt,D,nsteps,nparts)
        self.R = R
        pos0 = np.random.uniform(size = (nparts,2))
        pos0[:,0] = pos0[:,0]*2*np.pi # phi
        pos0[:,1] = np.arccos(2*pos0[:,1]-1) # theta
        
        # ---------- Calculation of positions-------------------
        x0, y0, z0=spherical2cart(R,pos0[:,0],pos0[:,1])
        self.xyz = np.concatenate((x0.reshape(-1,1),
                                  y0.reshape(-1,1),
                                  z0.reshape(-1,1)),axis=1)
        
        if self.save_coordinates:
            self.out_coords = np.zeros((nsteps,nparts,3))
            self.out_coords[0] = self.xyz
            
    def next_step(self):
        bm = np.random.normal(scale=np.sqrt(2*self.D*self.dt)/self.R,
                              size=(self.nparts,3))
        
        # norm xyz is R
        d_xyz = np.cross(self.xyz,bm)
        self.xyz+=d_xyz
        self.xyz = self.R*self.xyz/norm(self.xyz,axis=1)[:,np.newaxis]
        
        if self.save_coordinates:
            self.out_coords[self.current_frame]=self.xyz
        return self.xyz
    
    def get_positions(self):
        x,y,z=self.xyz[:,0], self.xyz[:,1], self.xyz[:,2]
        return x,y,z+self.R
    def parameters_dict(self):
        par_dict = super().parameters_dict()
        par_dict['R']= self.R
        return par_dict