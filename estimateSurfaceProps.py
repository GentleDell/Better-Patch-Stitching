#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:44:08 2020

@author: zhantao
"""

import torch
import torch.nn as nn 
from torch import Tensor


def getkNearestNeighbor( points : Tensor, k : int )  -> Tensor: 
    """
    It finds the k-nearest neighbors for each points. The mean of each point
    cluster is removed.

    Parameters
    ----------
    points : Tensor
        Point cloud to find the knn, [B, N, 3]. 
    k : int
        The number of nearest neighbors.

    Returns
    -------
    Tensor
        k nearest neighbors of each given points [B, N, k, 3].

    """
    numBatchs = points.shape[0]
    numPoints = points.shape[1]
    dimension = points.shape[2]
    
    distanceMatrix = ( points.detach().reshape(numBatchs, numPoints, 1, dimension)
                     - points.detach().reshape(numBatchs, 1, numPoints, dimension)
                      ).pow(2).sum(dim = 3).sqrt()
    
    colCloseIndice = distanceMatrix.topk(k, dim = 1, largest = False)[1].permute(0,2,1)
    kNearestPoints = points[:, colCloseIndice, :][torch.arange(numBatchs), torch.arange(numBatchs), :, :]
    
    # set the target points to be the origin of the local coordinates
    realignedPoint = kNearestPoints - points[:,:,None,:].repeat_interleave(k, dim = 2)
    
    return realignedPoint
    
    
def estimateNormal( kNearestPoints : Tensor ) -> Tensor:
    """
    It estimates the normal of each point according to the nearest neighbors. 

    Parameters
    ----------
    kNearestPoints : Tensor
        The k nearest neighbors of each point, [B, N, k, 3].

    Returns
    -------
    Tensor
        The normal of each point, [B, N, 3].

    """    
    # compute the covariance of each point set
    covarianc = kNearestPoints.permute(0, 1, 3, 2) @ kNearestPoints
    
    # eigen decompose each point set to get the eigen vector
    eigVector = covarianc.symeig( eigenvectors = True )[1][:,:,:,0]
    
    assert(eigVector.shape == torch.Size([kNearestPoints.shape[0], kNearestPoints.shape[1], 3]))
    
    return eigVector
    

def estimatePatchNormal(points : Tensor, numPatches : int, numNeighbor : int) -> Tensor:
    """
    It estimates patch-wise normal vectors.

    Parameters
    ----------
    points : Tensor
        Full point cloud, shape [B, N, 3].
    numPatches : int
        The number of patches.
    numNeighbor : int
        The number of nearest neigbors to estiamte normal.

    Returns
    -------
    normalPatchwise : Tensor
        Estimated patch-wise normal vectors.

    """
    
    batchSize  = int(points.shape[0])
    patchPoint = int(points.shape[1]/numPatches)
    normalVec  = []
    
    for batch in range(batchSize):
        
        batchNormalVec  = []
        
        for patchCnt in range(numPatches):
            
            patchPC = points[batch, patchCnt*patchPoint : (patchCnt + 1)*patchPoint, :][None, :, :]
            
            kNearest= getkNearestNeighbor(patchPC, numNeighbor)
            
            normals = estimateNormal(kNearest)
            
            batchNormalVec.append(normals)
        
        normalVec.append(torch.cat(batchNormalVec, dim = 1))
        
    normalPatchwise = torch.cat(normalVec, dim = 0)
    
    return normalPatchwise 


def estimateSurfVariance( kNearestPoints : Tensor ) -> Tensor:
    """
    It computes the surface variance for each points in the point cloud.
    
    Surface variance uses the rario between the minimum eigenvalue and the sum 
    of the eigen values.

    Parameters
    ----------
    kNearestPoints : Tensor
        The k nearest neighbors of each point, [B, N, k, 3].

    Returns
    -------
    Tensor
        The surface variance of each point, [B, N].

    """
    # compute the covariance of each point set
    covariance = kNearestPoints.permute(0, 1, 3, 2) @ kNearestPoints
    
    # eigen decompose each point set to get the eigen vector
    eigValues  = covariance.symeig( eigenvectors = True )[0]
    
    # surface variances
    surfaceVar = eigValues[:,:,0]/eigValues.sum(dim=2)
    
    assert(surfaceVar.shape == kNearestPoints.shape[:2])
    
    return surfaceVar


def estimatePatchSurfVar(points : Tensor, numPatches : int, numNeighbor : int) -> Tensor:
    """
    It estimates patch-wise surface variance.

    Parameters
    ----------
    points : Tensor
        Full point cloud.
    numPatches : int
        The number of patches.
    numNeighbor : int
        The number of nearest neigbors to estiamte surface variance.

    Returns
    -------
    SurfaceVarPatchwise : Tensor
        Estimated patch-wise surface variance.

    """
    
    batchSize  = int(points.shape[0])
    patchPoint = int(points.shape[1]/numPatches)
    patchVars  = []
    
    for batch in range(batchSize):
        
        batchSurfVars = []
        
        for patchCnt in range(numPatches):
            
            patchPC = points[batch, patchCnt*patchPoint : (patchCnt + 1)*patchPoint, :][None, :, :]
            
            kNearest= getkNearestNeighbor(patchPC, numNeighbor)
            
            surfVar = estimateSurfVariance(kNearest)
            
            batchSurfVars.append(surfVar)
        
        patchVars.append(torch.cat(batchSurfVars, dim = 1))
            
    SurfaceVarPatchwise = torch.cat(patchVars, dim = 0)
    
    return SurfaceVarPatchwise 


def lstq(A : Tensor, Y : Tensor) -> Tensor:
    """
    It finds the least square solution of the linear system Ax = Y to estimate
    surface parameters.

    We assume A to be full column rank, otherwise we would have to modify the
    corresponding submatrix by adding lambda * eye(num_col), where lambda ~ 0.01.
    
    Parameters
    ----------
    A : Tensor
        [N, k, 6].
    Y : Tensor
        [N, k, 1].

    Returns
    -------
    x : Tensor
        solution of the linear system, [N, 6, 1].

    """
    
    q, r = torch.qr(A)  # q: [N, k, l], r: [N, l, 6], l = min(k, 6)
    
    x = torch.inverse(r) @ q.permute(0,2,1) @ Y
    
    return x

    
def fitSurface( kNearestPoints : Tensor) -> Tensor:
    """
    It estimates the quadratic surface from the given k nearest niegbors of 
    each points.
    
    z = r(x, y) = a0x^2 + a1y^2 + a2xy + a3x + a4y + a5 = A.dot(a)

    Parameters
    ----------
    kNearestPoints : Tensor
        The k nearest neighbors of each point, [N, k, 3].

    Returns
    -------
    Tensor
        Estimated surface parameters, [N, 6].

    """
    numPoints = kNearestPoints.shape[0]
    numneighb = kNearestPoints.shape[1]
    
    x, y, z = kNearestPoints[:,:,0], kNearestPoints[:,:,1], kNearestPoints[:,:,2][:,:,None]
    
    A = torch.stack([x*x, y*y, x*y, x, y, torch.ones([numPoints, numneighb])], dim=2)
    a = lstq(A, z)[:,:,0]
    
    assert(a.shape == torch.Size([numPoints, 6]))
    
    return a


def estimateCurvature( points : Tensor, surfacePara : Tensor, normal : Tensor) -> Tensor:
    """
    It computes the curvature of the points, given the surface paras and normal.

    Parameters
    ----------
    points : Tensor
        Point cloud. [N, 3].
    surfacePara : Tensor
        Estimated quadratic surface parameters, [N, 6]..
    normal : Tensor
        DESCRIPTION.

    Returns
    -------
    Tensor
        Estimated surface curvatures, [N, 6, 2].

    """
    E = (2*surfacePara[:,0]*points[:,0] + surfacePara[:,2]*points[:,1] + surfacePara[:,3])**2 + 1 
    F = (2*surfacePara[:,0]*points[:,0] + surfacePara[:,2]*points[:,1] + surfacePara[:,3])  \
       *(2*surfacePara[:,1]*points[:,1] + surfacePara[:,2]*points[:,0] + surfacePara[:,4])
    G = (2*surfacePara[:,1]*points[:,1] + surfacePara[:,2]*points[:,0] + surfacePara[:,4])**2 + 1
    L = 2 * surfacePara[:,0][:,None] * normal
    M = surfacePara[:,2][:,None] * normal
    N = 2 * surfacePara[:,1][:,None] * normal
    
    div = E*G - F**2
    curve_Gaus = torch.sum(L*N - M**2, dim = 1)/div
    curve_mean = torch.sum(E[:,None]*N - 2*F[:,None]*M + G[:,None]*L, dim = 1)/div/2
    
    assert(curve_Gaus.shape == torch.Size([points.shape[0]]) and
           curve_mean.shape == torch.Size([points.shape[0]]))
    
    curvature = torch.stack([curve_Gaus, curve_mean], dim = 1)
    
    return curvature
    

def estimateDiffProp(points : Tensor, k : int) -> Tensor:
    """
    It estimates the differential properties for each point of the given point
    cloud, using k nearest neighbor. 

    Parameters
    ----------
    points : Tensor
        Point cloud. [N, 3].
    k : int
        The number of nearest neighbors.

    Returns
    -------
    Tensor
        The output point cloud, including points coordinate, normal vector and 
        curvatures, [N, 8].

    """
    kNearestPoints = getkNearestNeighbor(points, k)
    
    normalVectors  = estimateNormal(kNearestPoints)

    surfParameters = fitSurface(kNearestPoints)
    
    curvatureVects = estimateCurvature(points, surfParameters, normalVectors)
    
    outPointsCloud = torch.stack([points, normalVectors, curvatureVects])
    
    assert(outPointsCloud.shape == torch.Size([points.shape[0], 8]))

    return outPointsCloud


def chamferDistance( srcPoints : Tensor, refPoints : Tensor) -> Tensor:
    """
    It computes chamfer distance between the src points and ref points.

    Parameters
    ----------
    srcPoints : Tensor
        Source points.
    refPoints : Tensor
        Reference points.

    Returns
    -------
    Tensor
        Chamfer distance.

    """
    numBatchs = srcPoints.shape[0]
    numPoints = srcPoints.shape[1]
    dimension = srcPoints.shape[2]
    
    distanceMatrix = ( srcPoints.detach().reshape(numBatchs, numPoints, 1, dimension)
                     - refPoints.detach().reshape(numBatchs, 1, numPoints, dimension)
                      ).pow(2).sum(dim = 3).sqrt()
    
    srcNearestInd = distanceMatrix.argmin(dim = 2)
    refNearestInd = distanceMatrix.argmin(dim = 1)

    ref2SrcPoints = refPoints[:, srcNearestInd, :][torch.arange(numBatchs), torch.arange(numBatchs), :, :]
    src2RefPoints = srcPoints[:, refNearestInd, :][torch.arange(numBatchs), torch.arange(numBatchs), :, :]
    
    ref2SrcLoss = (srcPoints - ref2SrcPoints).pow(2).sum(dim=2)
    src2RefLoss = (refPoints - src2RefPoints).pow(2).sum(dim=2)

    chd = ref2SrcLoss.mean() + src2RefLoss.mean()
    
    return chd


class surfacePropLoss(nn.Module):
    
    def __init__(self, numPatches : int, kNeighbors : int, normals : bool = True,
                 normalLossAbs : bool = True, surfaceVariances : bool = False,
                 weight : list = [1,1]):
        
        nn.Module.__init__(self)
        self._numPatches = numPatches
        self._kNeighbors = kNeighbors
        self._useNormals = normals
        self._useSurfVar = surfaceVariances
                
        assert(len(weight) == 2 and "input weight has to contain 2 elements, the first for normal, the second to surfaceVar.")
        self._normalWeight  = weight[0]
        self._surfVarWeight = weight[1]
           
        self._normalLossAbs = normalLossAbs

        
    def forward(self, pointCloud : Tensor):
        
        kNearestNeighbor = getkNearestNeighbor(pointCloud, self._kNeighbors)
        surfacePropDiff  = []
        
        if self._useNormals:
            normalVecGlobal  = estimateNormal(kNearestNeighbor)
            normalPatchwise  = estimatePatchNormal(pointCloud, self._numPatches, self._kNeighbors)
            
            if self._normalLossAbs: 
                # use the absolute value difference between normal vectors
                normalVectorLoss = (normalPatchwise.abs() - normalVecGlobal.abs()).norm(dim=2).mean()[None]
            else:
                # direction insensitive normal loss
                normalVectorLoss = (1 - (normalPatchwise[:,:,None,:] @ normalVecGlobal[:,:,:,None]).squeeze().pow(2)).mean()[None]
            
            surfacePropDiff.append(normalVectorLoss * self._normalWeight)
            
        if self._useSurfVar:
            SurfVarGlobal    = estimateSurfVariance(kNearestNeighbor)
            SurfVarPatchwise = estimatePatchSurfVar(pointCloud, self._numPatches, self._kNeighbors)
            SurfVarianceLoss = (SurfVarPatchwise - SurfVarGlobal).pow(2).mean()[None]
        
            surfacePropDiff.append(SurfVarianceLoss * self._surfVarWeight)
                
        return surfacePropDiff