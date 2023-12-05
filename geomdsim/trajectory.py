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
    def __init__(self, dt,D,nsteps,nparts,*args, nr_save_coordinates=0, **kwargs):
        """Sets up the trajectories, in physical units. Units used: seconds, 
        micrometers."""
        super().__init__() # is this even useful?
        
        # these four are always there and therefore initialised in the abstract class
        self.dt = dt
        self.D = D
        self.nsteps = nsteps
        self.nparts = nparts
        self.nr_save_coordinates = nr_save_coordinates
        # insert code here for initial conditions
        self.xyz = (0,0,0)
        self.current_frame = 0
        self.bleaching_rate = None # default: no bleaching
        
        if self.nr_save_coordinates>0:
            self.out_coords = np.zeros((nsteps,self.nr_save_coordinates,3))
            
    def set_bleaching_rate(self,kb):
        self.bleaching_rate = kb

    def next_step(self):
        """Makes one step: updates values of all coordinates accordingly"""
        if self.nr_save_coordinates>0:
            self.out_coords[self.current_frame]=self.xyz[:self.nr_save_coordinates]
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
        
    def plot_trajectories(self, ncoords=None,ax=None):
        if self.nr_save_coordinates==0:
            return
        if ax is None:
            plt.figure()
            ax = plt.axes(projection='3d')
        # 
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')
        ax.set_zlabel('z [µm]')
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
    """Generator for trajectories on the surface of a Bacillus"""
    
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
    
    def next_step(self):
        super().next_step()
        bm = np.random.normal(scale=np.sqrt(2*self.D*self.dt)/self.R,
                              size=(self.nparts,3))

        vec_u = self.get_normal() # has norm R in physical coordinates
        
        d_xyz = np.cross(vec_u,bm)
        self.xyz = self.xyz+d_xyz
        vec_u = self.get_normal() # to normalise with norm of updated vector, slightly higher than R
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
        return x,y,z +self.R
    

class SphericalCytoTrajectory(Trajectory):
    def __init__(self, dt,D,nsteps,nparts,R,**kwargs):
        super().__init__(dt,D,nsteps,nparts,**kwargs)
        self.R = R
        
        
        pos0 = np.random.uniform(size = (nparts,3))
        pos0[:,0] = pos0[:,0]*2*np.pi # phi
        pos0[:,1] = np.arccos(2*pos0[:,1]-1) # theta
        pos0[:,2]*=R # r
        # ---------- Calculation of positions-------------------
        x0, y0, z0=spherical2cart(pos0[:,2],pos0[:,0],pos0[:,1])
        self.xyz = np.concatenate((x0.reshape(-1,1),
                                  y0.reshape(-1,1),
                                  z0.reshape(-1,1)),axis=1)
        
    def next_step(self):
        super().next_step()
        bm = np.random.normal(scale=np.sqrt(2*self.D*self.dt)/self.R,
                              size=(self.nparts,3))
        
        # norm xyz is R
        self.xyz+=bm
        
        normal = self.R*self.xyz/norm(self.xyz,axis=1)[:,np.newaxis]

        du = self.xyz-normal
        msk = (normal*du).sum(axis=1)>0
        self.xyz[msk]-=2*du[msk]
        return self.xyz
    
    def get_positions(self):
        x,y,z=self.xyz[:,0], self.xyz[:,1], self.xyz[:,2]
        return x,y,z+self.R
    
    def parameters_dict(self):
        par_dict = super().parameters_dict()
        par_dict['R']= self.R
        return par_dict
    
class BacillusCytoTrajectory(Trajectory):
    
    def __init__(self, dt,D,nsteps,nparts,R,length,**kwargs):
        super().__init__(dt,D,nsteps,nparts,**kwargs)
        self.R = R
        self.length = length
        # ---------- Distributes points on the spherical part----------------
        f1 = R/(R+2*length) # fraction on sphere
        pos0 = np.random.uniform(size = (int(f1*nparts),3))
        pos0[:,0] = pos0[:,0]*2*np.pi # phi
        pos0[:,1] = np.arccos(2*pos0[:,1]-1) # theta
        pos0[:,2]*=R # r
        # ---------- Calculation of positions-------------------
        x0, y0, z0=spherical2cart(pos0[:,2],pos0[:,0],pos0[:,1])
        
        # cylinder is on x axis
        x0[x0<0]-=length/2
        x0[x0>0]+=length/2
        
        # ----- Distributes points on the cylindrical part -----------------
        pos1 = np.random.uniform(size = (nparts-int(f1*nparts),3))
        pos1[:,0] = pos1[:,0]*2*np.pi # theta
        pos1[:,1] = pos1[:,1]*length-length/2 #x
        x1,y1,z1 = cylindrical2cart(R*pos1[:,2], pos1[:,0], pos1[:,1])
        
        # merges points on the sphere (x0) with points on the cylinder (x1)
        self.xyz = np.concatenate((np.concatenate((x0,x1)).reshape(-1,1),
                              np.concatenate((y0,y1)).reshape(-1,1),
                              np.concatenate((z0,z1)).reshape(-1,1) ),axis=1)
    
    def next_step(self):
        super().next_step()
        
        bm = np.random.normal(scale=np.sqrt(2*self.D*self.dt)/self.R,
                              size=(self.nparts,3))
        # scalar product
        spr = lambda v1,v2: np.sum(v1*v2,axis=1)
        
        """ 
        shapes: 
            xyz (nparts, 3)
            normal (nparts, 3)
            ratio (nparts)
            scalarproduct (nparts)
            msk (nparts)
            """
        # norm xyz is R
        xyz0=self.xyz.copy()
        normal_before = self.get_normal()
        self.xyz+=bm
        
        normal_after = self.get_normal()
       
        msk = norm(normal_after,axis=1)>self.R
        # dimensions: 
        # ratio of stuff still inside
        ratio=(self.R-norm(normal_before,axis=1))/(norm(normal_after,axis=1)-norm(normal_before,axis=1))
        self.xyz[msk] = (xyz0+bm*ratio[:,np.newaxis])[msk]
        normal_to_rebond = self.get_normal() # test: norm R OK
        normal_to_rebond/=norm(normal_to_rebond,axis=1)[:,np.newaxis]
        to_switch = (1-ratio[:,np.newaxis])*bm
        # takes the symmetric of remaining brownian motion with respect to normal at point of impact
        proj = spr(normal_to_rebond, to_switch)[msk]
        if proj.shape[0]>0:
            dbm = 2*proj[:,np.newaxis]*normal_to_rebond[msk]-to_switch[msk]
            self.xyz[msk]-=dbm
        return self.xyz
    
    def get_positions(self):
        x,y,z=self.xyz[:,0], self.xyz[:,1], self.xyz[:,2]
        return x,y,z+self.R
    
    def parameters_dict(self):
        par_dict = super().parameters_dict()
        par_dict['R'] = self.R
        return par_dict
    
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
        
        msk_neg = ~msk_pos
        vec_u[msk_neg,0] = np.min(
            np.concatenate(
                (vec_u[msk_neg,0].reshape(-1,1)+self.length/2,
                 np.zeros(np.count_nonzero(msk_neg)).reshape(-1,1)
                 )
                ,axis=1) # end concatenate
            ,axis=1)
        
        return vec_u
    