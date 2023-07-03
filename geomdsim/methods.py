#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:24:17 2023

@author: aurelienb
"""

import numpy as np
import datetime
import os

def spherical2cart(R,phi,theta):
    """Converts spherical coordinates to cartesian. Definition of spherical
    coordinates as described here: https://en.wikipedia.org/wiki/Spherical_coordinate_system"""
    x = R*np.cos(phi)*np.sin(theta)
    y = R*np.sin(phi)*np.sin(theta)
    z = R*np.cos(theta)
    return np.array([x, y, z])

def cylindrical2cart(R,theta,xcylindr):
    x = xcylindr
    y = R*np.cos(theta)
    z = R*np.sin(theta)
    return np.array([x, y, z])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    from SO: https://stackoverflow.com/questions/13685386/
    matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def prepare_savefolder(savepath):
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
        
    fname = datetime.datetime.now().__str__()[:19]
    savefolder = savepath+fname+"/"
    os.mkdir(savefolder)
    return savefolder