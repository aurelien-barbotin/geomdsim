#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:40:03 2023

@author: aurelienb

Methods used to simulate imaging on a TIRF microscope of individual particles
"""

import numpy as np

rng = np.random.default_rng()

class TIRF_Simulator(object):
    def __init__(self, npix_x, npix_y, psize, 
                 sigma_psf, dz_tirf, brightness, dt,
                 z_cutoff_factor):
        """Physical parameters have to be provided in physical units (e.g um).
        Parameters:
            npix_x (int): number of pixels in the final image in the x dimension.
            """
        self.npix_x = npix_x
        self.npix_y = npix_y
        self.psize = psize
        self.sigma_psf = sigma_psf
        self.dz_tirf = dz_tirf
        self.brightness = brightness
        self.dt = dt
        
        self.z_cutoff_factor = z_cutoff_factor
        
        # Arrays of coordinates in pixel space of 
        self.coords = np.meshgrid(np.arange(self.npix_x),np.arange(self.npix_y))
        self.frames_list = [] # where all frames will be saved

    def generate_frame(self, xyz):
        """From a list [x,y,z] of array coordinates of shape nparts, generates
        a TIRF image."""
    
        current_frame = np.zeros((self.npix_x, self.npix_y))
        x1, y1, z1 = xyz
        
        # conversion of xy to pixel coordinates. Shape of positions_new: (nparts, 2)
        positions_new = np.array([x1,y1]).T/self.psize
        positions_new+= (np.array([self.npix_x,self.npix_y])/2)[np.newaxis,:]
        
        znew = z1/self.psize #converts z to pixel coordinates
        
        # Everything is in pixel coordinates from here
        
        """msk0 = np.logical_and(
            positions_new>=0,
            positions_new<(np.array([self.npix_x,self.npix_y]))[np.newaxis,:]
                              ).all(axis=1)"""
        
        # anything above this along dimension z in pixel space is discarded 
        # because too dim to matter
        z_cutoff = self.z_cutoff_factor*self.dz_tirf/self.psize 
        # msk1 = np.logical_and(msk0,znew<z_cutoff)
        msk1 = znew<z_cutoff
        positions_new = positions_new[msk1,:]
        znew = znew[msk1]
        for k in range(positions_new.shape[0]):
            frame = self.coord2counts(positions_new[k,0], positions_new[k,1],znew[k])
            current_frame+=frame
        self.frames_list.append(frame)
        
        return frame
    
    def g2d(self,x0,y0,sigma):
        """2D Gaussian of std sigma at positions x0 and y0"""
        y,x=self.coords
        return np.exp(-( (x-x0)**2 + (y-y0)**2)/(2*sigma**2))/sigma**2
    
    def coord2counts(self,x,y,z):
        """Converts one 3D coordinate in the image of an object at said 
        coordinate in the TIRF field"""
        # pixel coordinates
        frame = self.g2d(x,y,self.sigma_psf/self.psize)*np.exp(-z*self.psize/self.dz_tirf)
        frame=rng.poisson(lam=frame* self.brightness*self.dt,size=frame.shape)
        return frame
    
    def get_stack(self):
        return np.array(self.frames_list)
    
    def parameters_dict(self):
            par_dict={"psize":self.psize,
                      "sigma_psf":self.sigma_psf,
                      "dz_tirf":self.dz_tirf,
                      "brightness":self.brightness,
                      "dt":self.dt,
                      "z_cutoff_factor":self.z_cutoff_factor}    
            return par_dict
