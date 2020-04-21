#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:35:01 2020

@author: zhantao
"""


import numpy as np
from numpy import array
import torch
from torch import Tensor

import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


clr_names = \
['aqua', 'blue', 'brown', 'chartreuse', 'chocolate', 'coral',
 'cornflowerblue', 'crimson', 'darkblue', 'darkcyan', 'darkgoldenrod',
 'darkgreen', 'darkmagenta', 'darkolivegreen', 'darkorange',
 'darkorchid', 'darkred', 'darkslateblue', 'darkturquoise',
 'darkviolet', 'deeppink', 'deepskyblue', 'firebrick', 'forestgreen',
 'gold', 'goldenrod', 'green', 'greenyellow', 'hotpink', 'indianred',
 'indigo', 'lawngreen', 'lightsalmon', 'lightseagreen', 'lime',
 'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue',
 'mediumorchid', 'mediumseagreen', 'mediumslateblue',
 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
 'midnightblue', 'navy', 'olive', 'orange', 'orangered', 'peru',
 'purple', 'rebeccapurple', 'red', 'royalblue', 'saddlebrown', 'salmon',
 'seagreen', 'sienna', 'slateblue', 'springgreen', 'steelblue', 'teal',
 'tomato', 'yellow', 'yellowgreen']


def generateColors(numPatches : float, numPoints : float) -> array:
    """
    It generates colors for different patches.

    Parameters
    ----------
    numPatches : float
        The number of patches.
    numPoints : float
        The number of points in the point cloud.

    Returns
    -------
    clrPatches : array
        Array of color, [numPoints, 3], every numPoints/numPatches rows have a difference color.

    """
    clrPatches = np.ones([numPoints, 3])
    ptPerPatch = int(numPoints / numPatches)
    
    for cnt in range(numPatches):
        clrPatches[cnt*ptPerPatch : (cnt+1)*ptPerPatch, :] = mcolors.to_rgb(mcolors.CSS4_COLORS[clr_names[cnt]])
        
    return clrPatches


def custom_draw_geometry_with_key_callback(pcd):
    """
    Call back function to make the point cloud interactable.

    Parameters
    ----------
    pcd : TYPE
        Point cloud in open3d.

    Returns
    -------
    None.

    """
    
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "./renderoption.json")
        return False

    def increasePoints(vis):
        opt = vis.get_render_option()
        opt.point_size += 0.5 
        return False
    
    def decreasePoints(vis):
        opt = vis.get_render_option()
        opt.point_size -= 0.5 
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord("+")] = increasePoints
    key_to_callback[ord("-")] = decreasePoints
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries_with_key_callbacks([pcd, mesh_frame], key_to_callback)


def visPointCloud( points : Tensor, color : Tensor = None, absColor : bool = False):
    """
    It visualizes the given point cloud with optional color.

    Parameters
    ----------
    points : Tensor
        Point cloud to visualized the noraml, [N, 3].
    color  : Tensor, optional
        color vectors, [N, 3]. The default is None.
    absColor : bool, optional
        Whether to show the absolute value of the color vectors. The default is False.

    Returns
    -------
    None.

    """
    
    assert(len(points.shape) == 2 and points.shape[1] == 3 and 'The shape of point cloud should be [N, 3]')
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector( points )
    
    if color is not None:
        
        assert(len(points.shape) == 2 and points.shape[1] == 3 and 'The shape of colors should be [N, 3]')
        
        if absColor:
            viscolor = color.clone().abs()
        elif color.min().item() < 0:
            viscolor = (color + 1)/2
        else:
            viscolor = color.clone()
    
        pcd.colors = o3d.utility.Vector3dVector( viscolor )
    
    custom_draw_geometry_with_key_callback(pcd)


def visNormalDiff( points : Tensor, globalNormal : Tensor, patchwiseNormal : Tensor, 
                   numPatches : int, hAlign : bool = True, visDiff : bool = False, 
                   visAbs : bool = True):
    """
    It visulizes the global normal and patchwise normal. 

    Parameters
    ----------
    points : Tensor
        Points cloud to visualize the normal.
    globalNormal : Tensor
        Global normal vector to be visualized.
    patchwiseNormal : Tensor
        Patchwise normal vector to be visualized
    numPatches : int
        Number of patches.
    hAlign : bool, optional
        Whether to align the normal+points horizontally in the figure. The default is True.
    visDiff : bool, optional
        Whether to visualize the difference between patchwise and global normal. The default is False.
    visAbs : bool, optional
        Whether to visualize the absolute value of normal. The default is True.

    Returns
    -------
    None.

    """
    
    # patch color
    patchColors = generateColors(numPatches, points.shape[0])
    patchColors = torch.from_numpy(patchColors).float()
    
    # separate points
    pcPatchwise, pcGlobal = points.clone(), points.clone()
    if hAlign:
        pcGlobal[:,0] = pcGlobal[:,0] + 2
        pcPatchwise[:,0] = pcPatchwise[:,0] - 2
    else:
        pcGlobal[:,1] = pcGlobal[:,1] + 1
        pcPatchwise[:,1] = pcPatchwise[:,1] - 1
    
    # stack points and normal
    if visDiff:
        pcStacked = torch.cat([points, pcPatchwise], dim = 0)
        clstacked = torch.cat([patchColors, patchwiseNormal - globalNormal], dim = 0)
    else:
        pcStacked = torch.cat([points, pcPatchwise, pcGlobal], dim = 0)
        clstacked = torch.cat([patchColors, patchwiseNormal, globalNormal], dim = 0)
    
    visPointCloud(pcStacked, clstacked, visAbs)


def visNSurface(parameters : Tensor, points : Tensor):
    """
    It visualizes the fitted surface of the given parameters.

    Parameters
    ----------
    parameters : Tensor
        The parameters to fit the surface.
    points : Tensor
        The points being fitted.

    Returns
    -------
    None.

    """
    x, y = np.meshgrid(np.linspace(points[:,0].min(), points[:,0].max(), num = 20),
                       np.linspace(points[:,1].min(), points[:,1].max(), num = 20))
    x, y = x.flatten(), y.flatten()
    A = np.stack([x*x, y*y, x*y, x, y, np.ones([400])], axis=1)
    z = A@parameters.numpy()[:,None]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.scatter(x,y,z)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()



# ========== using ipyvolume with jupyter notebook ==========

# import torch
# import numpy as np
# import ipyvolume as ipv
# from torch import Tensor
# import open3d as o3d
# import matplotlib.colors as mcolors
# from scipy.spatial.transform import Rotation


# from visutils import generateColors
# from estimateSurfaceProps import *

# pointCloud = torch.load('./ShapeReconstructionNet/data/reconstructedShape/pc1.pt')[0][None, :, :]

# optPoints  = torch.load('JointOpt_pc_directioninsensitive.pt')
# patchColor = generateColors(numPatches=25, numPoints=pointCloud.shape[1])
# x, y, z    = optPoints.numpy()[:,0], optPoints.numpy()[:,1], optPoints.numpy()[:,2]

# ipv.figure(width=600, height=600)
# ipv.pylab.scatter(x+0.5, y, z, color = patchColor, size=4, marker="sphere")
# ipv.squarelim()
# ipv.show()