#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:02:41 2020

Some pytorch3D render codes are from the Facebook pytorch3D project.

@author: zhantao
"""

import torch 
from torch import Tensor
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights,     
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardPhongShader
)

from visutils import generateColors


def image_grid( images, rows=None, cols=None, fill: bool = True, 
                show_axes: bool = False, rgb: bool = True):
    """
    # Copyright (c) Facebook, Inc. and its affiliates. All rights reserved. 
    This function ONLY.
    
    A util function for plotting a grid of images.

    Args:
        images: 
            (N, H, W, 4) array of RGBA images
        rows:
            number of rows in the grid
        cols:
            number of columns in the grid
        fill:
            boolean indicating if the space between images should be filled
        show_axes:
            boolean indicating if the axes of the plots should be visible
        rgb:
            boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1
        
    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        ax.figsize = (5,5)
        
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()


def genTrifacet( numPoints: int, numPatch: int ) -> Tensor:
    '''
    It generates the triangular facet connections for point clouds generated 
    from the regularly sampled uv points.
    
    See FNSamplerRegularGrid().

    Parameters
    ----------
    numPoints : int
        The number of points in a point cloud.
    numPatch : int
        The number of the patches in a point cloud.

    Returns
    -------
    Tensor
        triangular facets of for the point cloud.

    '''
    
    patchSize = Tensor([numPoints/numPatch])
    rowSize   = torch.sqrt(patchSize)
    pointList = torch.arange(patchSize.item())
    
    leftMost  = pointList%rowSize == 0
    rightMost = torch.roll( leftMost, rowSize.int().item()-1, 0)
    bottom    = pointList >= (patchSize - rowSize)
    
    topLeftTri= pointList[~rightMost * ~bottom][:,None]
    bottRigTri= pointList[~leftMost * ~bottom][:, None]
    
    triFacetsPatch = torch.cat(
                        [torch.cat([topLeftTri, topLeftTri+1, topLeftTri+rowSize ], dim = 1),
                         torch.cat([bottRigTri, bottRigTri+rowSize, bottRigTri+rowSize-1 ], dim = 1)],
                         dim = 0)
    
    increments  = torch.arange(0, numPoints, patchSize.item()).repeat_interleave( triFacetsPatch.shape[0] )[:, None]
    triFacetsPc = triFacetsPatch.repeat(numPatch, 1) + increments
    
    return triFacetsPc
    

def createRenderer( device, camera, light, imageSize ):
    '''
    It creates a pytorch3D renderer with the given camera pose, light source
    and output image size.

    Parameters
    ----------
    device : 
        Device on which the renderer is created.
    camera : 
        Camera pose.
    light  : 
        Position of the light source.
    imageSize : 
        The size of the rendered image.

    Returns
    -------
    renderer : 
        Pytorch3D renderer.

    '''
    if camera is None:
        camera = (2.0, -20.0, 180.0)
    if light is None:
        light  = (0.0, 2.0, 0.0)
    
    # Initialize an OpenGL perspective camera.
    # With world coordinates +Y up, +X left and +Z into the screen. 
    R, T = look_at_view_transform(camera[0], camera[1], camera[2]) 
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
    
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=imageSize, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    
    # Place a point light at -y direction. 
    lights = PointLights(device=device, location=[[ light[0], light[1], light[2] ]])
    
    # Create a phong renderer by composing a rasterizer and a shader. 
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    
    return renderer


def renderMultiPose( mesh, render, batch_size = 20, row: int = 4, 
                     col: int = 5, device = 'cuda' ):
    '''
    It generates multiple camera poses to render images with settings in 
    the given renderer. The number of poses is the batch_size.

    Parameters
    ----------
    mesh : 
        Mesh to be rendered.
    render : 
        Pytorch3D renderer.
    batch_size : optional
        The number of camera poses to be generated. The default is 20.
    row : int, optional
        The number of rows when show the images. The default is 4.
    col : int, optional
        The number of columns when show the images. The default is 5.
    device : optional
        The device where the rendering takes place. The default is 'cuda'.

    Returns
    -------
    None.

    '''
    
    # Create a batch of meshes by repeating the cow mesh and associated textures. 
    # Meshes has a useful `extend` method which allows us do this very easily. 
    # This also extends the textures. 
    meshes = mesh.extend(batch_size)
    
    # Get a batch of viewing angles. 
    elev = torch.linspace(0, 180, batch_size)
    azim = torch.linspace(-180, 180, batch_size)
    
    # All the cameras helper methods support mixed type inputs and broadcasting. So we can 
    # view the camera from the same distance and specify dist=2.7 as a float,
    # and then specify elevation and azimuth angles for each viewpoint as tensors. 
    R, T = look_at_view_transform(dist=2.0, elev=elev, azim=azim)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
    
    # We can pass arbirary keyword arguments to the rasterizer/shader via the renderer
    # so the renderer does not need to be reinitialized if any of the settings change.
    images = render(meshes, cameras=cameras)
    
    image_grid(images.cpu().numpy(), rows=row, cols=col, rgb=True)


def renderPointcloud( pathToPc: str, numPatch: int, camera: tuple = None, 
                      light: tuple = None, imageSize : int = 512, 
                      batchSize: int = 20, row : int = 4, col: int = 5 ):
    '''
    It renders the point clouds under the given path with the given parameters.
    The point clouds have to be generated from regularly sampled uv points.

    Parameters
    ----------
    pathToPc : str
        Path to the point clouds that are generated from regularly sampled
        2D uv points.
    numPatch : int
        The number of patches contained in each point cloud.
    camera : tuple, optional
        Camera pose to render the image. The default is None.
    light : tuple, optional
        Light source position to render the image. The default is None.
    imageSize : int, optional
        The size of the rendered image. The default is 512.
    batchSize : int, optional
        The number of multiple poses to be rendered. The default is 20.
    row : int, optional
        The number of rows when show the images. The default is 4.
    col : int, optional
        The number of columns when show the images. The default is 5.

    Returns
    -------
    None.

    '''
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    vertex  = torch.load(pathToPc).to(device)
    facets  = genTrifacet(vertex.shape[1], numPatch).to(device)
    render  = createRenderer( device, camera, light, imageSize )
    
    vertRGB = torch.tensor(generateColors(numPatch, vertex.shape[1])[None]).to(device)
    texture = Textures(verts_rgb=vertRGB)
    
    for ct in range(vertex.shape[0]):       
        
        triMesh = Meshes(verts=[vertex[ct]], faces=[facets], textures=texture)
        
        if camera is not None:
            images = render(triMesh)
            
            plt.figure(figsize = [10, 10])
            plt.imshow(images.cpu().numpy()[0])
        else:
            if row*col == batchSize:
                renderMultiPose(triMesh, render, batchSize, row, col)
            else:
                renderMultiPose(triMesh, render, 20, 4, 5)  
    

renderPointcloud('../../models/comparison/plane/plane_fromScratch_100_None/prediction/regularSample1.pt', 25,
                 camera = (1.2, 20.0, 180.0), light = (0.0, 2.0, 0.0))
