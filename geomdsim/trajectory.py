#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:07:17 2023

@author: aurelienb

objects uused to described trajectories in different geometrical configurations
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm



from geomdsim.methods import spherical2cart, set_axes_equal, cylindrical2cart

# 1/ trajectory simulation: 1/ generate initial conditions 2/ moves these conditions
# 2/ from trajectory to image
# 2/ process the resulting stack, generate folder etc

class Trajectory(object):
    def __init__(self, dt,D,nsteps,nparts,*args, save_coordinates=False, **kwargs):
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
        
        if self.save_coordinates:
            self.out_coords = np.zeros((nsteps,nparts,3))
            
    def next_step(self):
        """Makes one step: updates values of all coordinates accordingly"""
        if self.save_coordinates:
            self.out_coords[self.current_frame]=self.xyz
        self.current_frame+=1
    def get_positions(self):
        """Method to get the positions recentered for imaging. Recentring depends
        on the simulaiton method, e.g in a sphere you want to add the radius to the z coordinates
        so that we simulate imaging bottom of the sphere"""
        raise NotImplementedError("get_positions method not implemented for this class")
    
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
        
    def plot_trajectories(self, ncoords=None):
        if not self.save_coordinates:
            return
        plt.figure()
        ax = plt.axes(projection='3d')
        # 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if ncoords is None:
            ncoords = self.out_coords.shape[1]
        for j in range(ncoords):
            ax.plot3D(self.out_coords[:,j,0],self.out_coords[:,j,1], 
                      self.out_coords[:,j,2])
        set_axes_equal(ax)
        
    def parameters_dict(self):
        par_dict= {"dt":self.dt,
                   "D":self.D,
                   "nsteps":self.nsteps,
                   "nparts":self.nparts}
        return par_dict

class SphericalTrajectory(Trajectory):
    def __init__(self, dt,D,nsteps,nparts,R,**kwargs):
        super().__init__(dt,D,nsteps,nparts,**kwargs)
        self.R = R
        pos0 = np.random.uniform(size = (nparts,2))
        pos0[:,0] = pos0[:,0]*2*np.pi # phi
        pos0[:,1] = np.arccos(2*pos0[:,1]-1) # theta
        
        # ---------- Calculation of positions-------------------
        x0, y0, z0=spherical2cart(R,pos0[:,0],pos0[:,1])
        self.xyz = np.concatenate((x0.reshape(-1,1),
                                  y0.reshape(-1,1),
                                  z0.reshape(-1,1)),axis=1)
    
            
    def next_step(self):
        super().next_step()
        bm = np.random.normal(scale=np.sqrt(2*self.D*self.dt)/self.R,
                              size=(self.nparts,3))
        
        # norm xyz is R
        d_xyz = np.cross(self.xyz,bm)
        self.xyz+=d_xyz
        self.xyz = self.R*self.xyz/norm(self.xyz,axis=1)[:,np.newaxis]

        return self.xyz
    
    def get_positions(self):
        x,y,z=self.xyz[:,0], self.xyz[:,1], self.xyz[:,2]
        return x,y,z+self.R
    
    def parameters_dict(self):
        par_dict = super().parameters_dict()
        par_dict['R']= self.R
        return par_dict
    
class BacillusTrajectory(Trajectory):
    
    def __init__(self, dt,D,nsteps,nparts,R, length,**kwargs):
        super().__init__(dt,D,nsteps,nparts,**kwargs)
        self.R = R
        self.length = length
    
        # ---------- Distributes points on the spherical part----------------
        f1 = R/(R+2*length) # fraction on sphere
        pos0 = np.random.uniform(size = (int(f1*nparts),2))
        # phi
        pos0[:,0] = pos0[:,0]*2*np.pi
        # theta
        pos0[:,1] = np.arccos(2*pos0[:,1]-1)
        
        x0, y0, z0=spherical2cart(R,pos0[:,0],pos0[:,1])
        # cylinder is on x axis
        x0[x0<0]-=length/2
        x0[x0>0]+=length/2
        
        # ----- Distributes points on the cylindrical part -----------------
        pos1 = np.random.uniform(size = (nparts-int(f1*nparts),2))
        pos1[:,0] = pos1[:,0]*2*np.pi # theta
        pos1[:,1] = pos1[:,1]*length-length/2 #x
        x1,y1,z1 = cylindrical2cart(R, pos1[:,0], pos1[:,1])
        
        # merges points on the sphere (x0) with points on the cylinder (x1)
        self.xyz = np.concatenate((np.concatenate((x0,x1)).reshape(-1,1),
                              np.concatenate((y0,y1)).reshape(-1,1),
                              np.concatenate((z0,z1)).reshape(-1,1) ),axis=1)
        if self.save_coordinates:
            self.out_coords = np.zeros((nsteps,nparts,3))
            self.out_coords[0] = self.xyz
    
    def next_step(self):
        super().next_step()
        bm = np.random.normal(scale=np.sqrt(2*self.D*self.dt)/self.R,
                              size=(self.nparts,3))

        vec_u = self.get_normal() # has norm R in physical coordinates
        
        d_xyz = np.cross(vec_u,bm)
        self.xyz = self.xyz+d_xyz
        norm_u=norm(vec_u,axis=1)[:,np.newaxis]
        self.xyz = self.xyz + (self.R-norm_u)*vec_u/norm_u
        
    def get_normal(self):
        """Calculates a vector normal to a point in a bacillus configuration"""
        vec_u = self.xyz.copy()
        # positive values of x
        msk_pos = vec_u[:,0]>0
        vec_u[msk_pos,0] = np.max(
            np.concatenate(
                (
                    vec_u[msk_pos,0].reshape(-1,1)-self.length/2,
                    np.zeros(np.count_nonzero(msk_pos)
                          ).reshape(-1,1))
                ,axis=1)
            ,axis=1)
        
        msk_neg = vec_u[:,0]<=0
        vec_u[msk_neg,0] = np.min(
            np.concatenate(
                (vec_u[msk_neg,0].reshape(-1,1)+self.length/2,
                 np.zeros(np.count_nonzero(msk_neg)).reshape(-1,1)
                 )
                ,axis=1) # end concatenate
            ,axis=1)
        return vec_u
    
    def parameters_dict(self):
        par_dict = super().parameters_dict()
        par_dict['R']= self.R
        par_dict['length'] = self.length
        return par_dict
    
    def get_positions(self):
        x,y,z=self.xyz[:,0], self.xyz[:,1], self.xyz[:,2]
        return x,y,z+self.R