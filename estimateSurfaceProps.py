#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:44:08 2020

@author: zhantao
"""

import torch
import torch.nn as nn 
from torch import Tensor

from visutils import visSpecifiedPoints, visNormalDiff

_USRINF = 1e8

def getkNearestNeighbor( points : Tensor, k : int, gtPoints : Tensor, 
                         gtNormal : Tensor, angleThreshold : float)  -> Tensor: 
    """
    It finds the k-nearest neighbors for each points. The mean of each point
    cluster is removed.

    Parameters
    ----------
    points : Tensor
        Point cloud to find the knn, [B, N, 3]. 
    k : int
        The number of nearest neighbors.
    gtPoints : Tensor
        The ground truth point cloud. It is optional.
    gtNormal : Tensor
        The ground truth normal of the point cloud.
        
        To avoid including a point from other sides in KNN, making the normal
        and surface variance estimation fail/do not help the final point cloud,
        the gtNormal of the point's correspoing gtPoint are used. Only points
        whose gtNormal is close the noraml of the src point will be considered
        as KNN of the targe point. 
        
        It is optional but it has to be given together with gtPoints. 
    angleThreshold : float 
        The threshold to remove bad knn points.

    Returns
    -------
    Tensor
        k nearest neighbors of each given points [B, N, k, 3].

    """
    srcnumBatchs = points.shape[0]
    srcnumPoints = points.shape[1]
    srcdimension = points.shape[2]
    
    distanceMatrix = ( points.detach().reshape(srcnumBatchs, srcnumPoints, 1, srcdimension)
                     - points.detach().reshape(srcnumBatchs, 1, srcnumPoints, srcdimension)
                      ).pow(2).sum(dim = 3).sqrt()
    
    # if the gtNormal and gtPoints are available, we would use the gtNormal to 
    # constraint the knn with the angle threshold
    if gtNormal is not None and gtPoints is not None:
        
        dstnumBatchs = gtPoints.shape[0]
        dstnumPoints = gtPoints.shape[1]
        dstdimension = gtPoints.shape[2]
    
        # get the normal of the nearest groundtruth point of each point
        distanceToGT  = ( points.detach().reshape(srcnumBatchs, srcnumPoints, 1, srcdimension)
                          - gtPoints.detach().reshape(dstnumBatchs, 1, dstnumPoints, dstdimension)
                          ).pow(2).sum(dim = 3).sqrt()
        srcNearestInd = distanceToGT.argmin(dim = 2)
        srcIdealNormal= gtNormal[:, srcNearestInd, :][torch.arange(srcnumBatchs), torch.arange(srcnumBatchs), :, :]
        
        angleBtnormal = srcIdealNormal @ srcIdealNormal.permute(0,2,1) 
        invalidKnnPt  = angleBtnormal <= torch.tensor(angleThreshold).cos() 
        distanceMatrix[invalidKnnPt] = _USRINF
        
        # This call could be used to verify that the global knn is 
        # constrainted by the angle threshold 
        # visSpecifiedPoints(points[0].detach(), [torch.where(invalidKnnPt[0,769] == 0)[0]])
        
    # if the gtNormal is given while the gtPoints is not given,it treats 
    # the gtNormal as the predicted normal vector, so here we directly use 
    # the normal to reject invalid neighbors.
    elif gtNormal is not None and gtPoints is None:
        predictNormal = gtNormal/gtNormal.norm(dim=2)[:,:,None]
        angleBtnormal = predictNormal @ predictNormal.permute(0,2,1) 
        invalidKnnPt  = angleBtnormal <= torch.tensor(angleThreshold).cos() 
        distanceMatrix[invalidKnnPt] = _USRINF
        
        # This call could be used to verify that the global knn is 
        # constrainted by the angle threshold 
        # visSpecifiedPoints(points[0].detach(), [torch.where(invalidKnnPt[0,769] == 0)[0]])
        
    colCloseIndice = distanceMatrix.topk(k, dim = 1, largest = False)[1].permute(0,2,1)
    kNearestPoints = points[:, colCloseIndice, :][torch.arange(srcnumBatchs), torch.arange(srcnumBatchs), :, :]
    
    # This call could be used to verify that patchwise knns are not 
    # constrainted by the angle threshold.
    # visSpecifiedPoints(points[0].detach(), [colCloseIndice[0,69]])
    
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
    

def estimatePatchNormal(points : Tensor, numPatches : int, numNeighbor : int, 
                        gtPoints : Tensor = None, gtNormal : Tensor = None,
                        angleThreshold : float = 1.5708) -> Tensor:
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
    gtPoints : Tensor
        The ground truth point cloud. It is optional.
    gtNormal : Tensor
        The ground truth normal of the point cloud.
        
        To avoid including a point from other sides in KNN, making the normal
        and surface variance estimation fail/do not help the final point cloud,
        the gtNormal of the point's correspoing gtPoint are used. Only points
        whose gtNormal is close the noraml of the src point will be considered
        as KNN of the targe point. 
        
        It is optional but it has to be given together with gtPoints. 
    angleThreshold : float 
        The threshold to remove bad knn points.

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
            
            if gtPoints is not None and gtNormal is not None:
                kNearest = getkNearestNeighbor(patchPC, numNeighbor, gtPoints[batch][None, :, :],
                                              gtNormal[batch][None, :, :], angleThreshold)
            else:
                kNearest = getkNearestNeighbor(patchPC, numNeighbor, gtPoints, gtNormal, angleThreshold)
                
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


def estimatePatchSurfVar(points : Tensor, numPatches : int, numNeighbor : int,
                         gtPoints : Tensor = None, gtNormal : Tensor = None,
                         angleThreshold : float = 1.5708) -> Tensor:
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
            
            if gtPoints is not None and gtNormal is not None:
                kNearest = getkNearestNeighbor(patchPC, numNeighbor, gtPoints[batch][None, :, :],
                                              gtNormal[batch][None, :, :], angleThreshold)
            else:
                kNearest = getkNearestNeighbor(patchPC, numNeighbor, gtPoints, gtNormal, angleThreshold)
            
            surfVar = estimateSurfVariance(kNearest)
            
            batchSurfVars.append(surfVar)
        
        patchVars.append(torch.cat(batchSurfVars, dim = 1))
            
    SurfaceVarPatchwise = torch.cat(patchVars, dim = 0)
    
    return SurfaceVarPatchwise 


def lstq(A : Tensor, Y : Tensor) -> Tensor:
    """
    It finds the least square solution of the linear system Ax = Y to estimate
    surface parameters.gtPoints : Tensor
        The ground truth point cloud. It is optional.
    gtNormal : Tensor
        The ground truth normal of the point cloud.
        
        To avoid including a point from other sides in KNN, making the normal
        and surface variance estimation fail/do not help the final point cloud,
        the gtNormal of the point's correspoing gtPoint are used. Only points
        whose gtNormal is close the noraml of the src point will be considered
        as KNN of the targe point. 
        
        It is optional but it has to be given together with gtPoints. 

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


def criterionStitching(uvspace: Tensor, points: Tensor, numPatch: int, 
                       marginSize: float = 0.1):
    '''
    It computes the stitching criterion to evaluate the patch stitching 
    quality.
    
    Given a predicted point clouds batch, for each patch of each point cloud,
    it takes points at boundaries, according to the marginSize parameter. Then
    it compute the distance between each point of an margine to the points of
    margins of other patches. Next, for each margin area, it averages the
    smallest distance of points in the margin as the stitching quality of the
    margin. Finally, it averages the stitching quality of all margins of all 
    patches in this batch as the stitching quality.

    Parameters
    ----------
    uvspace : Tensor
        The sample 2D points in the uv space, [B, N, 2].
    points : Tensor
        The predicted point cloud, [B, N, 3].
    numPatch : int
        The number of patches.
    marginSize : float, optional
        The size of margines. The default is 0.1.

    Returns
    -------
    Tensor
        stitching quality.

    '''
    
    batchSize  = points.shape[0]
    patchSize  = points.shape[1]/numPatch
    marginSize = min(max(marginSize, 0.01), 0.5)
    
    # get all points at different margins according to the uv points
    leftMargin = (uvspace[:,:,0] < marginSize) * (uvspace[:,:,1] < 1 - marginSize)
    topMargin  = (uvspace[:,:,0] < 1 - marginSize) * (uvspace[:,:,1] > 1 -  marginSize)
    rightMargin= (uvspace[:,:,0] > 1 - marginSize) * (uvspace[:,:,1] > marginSize)
    bottMargin = (uvspace[:,:,0] > marginSize) * (uvspace[:,:,1] < marginSize)
    
    # to visualize the margin points and the full point cloud
    # visSpecifiedPoints(points[0].detach().to('cpu'),
    #                     [torch.where(leftMargin[0])[0].to('cpu'),
    #                     torch.where(topMargin[0])[0].to('cpu'),
    #                     torch.where(rightMargin[0])[0].to('cpu'),
    #                     torch.where(bottMargin[0])[0].to('cpu')])
    
    batchStitchingLoss = torch.tensor([0.])
    batchStitchingDist = []
    batchStitchingIndex= []
    for batch in range(batchSize):
        # points at the left margin, xxPatch is to distinguish patches and 
        # margins, for possible processing.
        leftPoints = points[batch, leftMargin[batch,:], :]
        leftPatch  = torch.where(leftMargin[batch,:])[0]//patchSize + 0.1
        # points at the top margin
        topPoints  = points[batch, topMargin[batch,:], :]
        topPatch   = torch.where(topMargin[batch,:])[0]//patchSize + 0.2
        # points at the right margin
        rightPoints = points[batch, rightMargin[batch,:], :]
        rightPatch  = torch.where(rightMargin[batch,:])[0]//patchSize + 0.3
        # points at the bottom margin
        bottPoints = points[batch, bottMargin[batch,:], :]
        bottPatch  = torch.where(bottMargin[batch,:])[0]//patchSize + 0.4
        
        boundaryPoints = torch.cat((leftPoints, topPoints, rightPoints, bottPoints), dim = 0)
        boundaryIndice = torch.cat((leftPatch, topPatch, rightPatch, bottPatch), dim = 0)
        
        # sort the points and indices by patches and by margins
        # 0.1 = left, 0.2 = top, 0.3 = right, 0.4 = bottom
        patchOrder = torch.argsort(boundaryIndice)
        boundaryPoints = boundaryPoints[patchOrder, :]
        boundaryIndice = boundaryIndice[patchOrder]
        
        # visSpecifiedPoints(points[batch].detach().to('cpu'), [torch.where(boundaryIndice < 1)[0].to('cpu')])
        
        # compute distances between points in margins
        distanceMatrix = (boundaryPoints[None, :, :] - boundaryPoints[:, None, :]).norm(dim=2)
        
        # remove distances between point from the same patch
        for ind in range(numPatch):
            subMatrix = torch.where((boundaryIndice < ind + 1) * (boundaryIndice > ind))[0]
            distanceMatrix[subMatrix.min():subMatrix.max(), subMatrix.min():subMatrix.max()] = _USRINF
        
        # keep the smallest distance 
        smallestDistance = torch.min(distanceMatrix, dim=1)[0]
        
        # keep the distance and indices for future use
        batchStitchingDist.append(smallestDistance)
        batchStitchingIndex.append(boundaryIndice)
        
        # compute stitching loss for each patch
        # for each margin of the patch, compute the average smallest. Then sum
        # the 4 averages up as the patch stitching loss
        for ind in range(numPatch):
            batchStitchingLoss += smallestDistance[boundaryIndice == ind + 0.1].mean() + \
                                  smallestDistance[boundaryIndice == ind + 0.2].mean() + \
                                  smallestDistance[boundaryIndice == ind + 0.3].mean() + \
                                  smallestDistance[boundaryIndice == ind + 0.4].mean()
    
    # average the loss over all batches
    batchStitchingLoss /= batchSize
    
    return batchStitchingLoss
        

def normalDifference(gtPoints: Tensor, gtNormals: Tensor, 
                     predPoints: Tensor, globalNormals: Tensor):
    '''
    It computes the difference of the normal vectors.
    
    For each point, it search for the closest gt point and uses the gt normal 
    of the gt point as the ideal normal of the point. Then, it compute the 
    orientation insensitive difference between the ideal normal vectors and 
    the globally estimated normal vector.

    Parameters
    ----------
    gtPoints : Tensor
        Ground truth points.
    gtNormals : Tensor
        Ground truth normals.
    predPoints : Tensor
        Predicted points.
    globalNormals : Tensor
        Globally estimated normals.

    Returns
    -------
    normalDiffAvg : 
        The orientation insensitive difference.

    '''
    gtNormals = gtNormals/torch.norm(gtNormals, dim = 2)[:,:,None]
    
    srcnumBatchs = predPoints.shape[0]
    srcnumPoints = predPoints.shape[1]
    srcdimension = predPoints.shape[2]
    
    dstnumBatchs = gtPoints.shape[0]
    dstnumPoints = gtPoints.shape[1]
    dstdimension = gtPoints.shape[2]

    # get the normal of the nearest groundtruth point of each point
    distanceToGT  = ( predPoints.detach().reshape(srcnumBatchs, srcnumPoints, 1, srcdimension)
                      - gtPoints.detach().reshape(dstnumBatchs, 1, dstnumPoints, dstdimension)
                      ).pow(2).sum(dim = 3).sqrt()
    srcNearestInd = distanceToGT.argmin(dim = 2)
    srcIdealNormal= gtNormals[:, srcNearestInd, :][torch.arange(srcnumBatchs), torch.arange(srcnumBatchs), :, :]
    
    # use direction insensitive criterion
    normalDiffAvg = (1 - (srcIdealNormal[:,:,None,:] @ globalNormals[:,:,:,None]).squeeze().pow(2)).mean()
    
    # visualize the normal difference
    # visNormalDiff(predPoints[0].to('cpu'), srcIdealNormal[0].to('cpu'), globalNormals[0].to('cpu'), 25)
    
    return normalDiffAvg


class surfacePropLoss(nn.Module):
    
    def __init__(self, numPatches : int, kNeighbors : int, normals : bool = True,
                 normalLossAbs : bool = True, surfaceVariances : bool = False,
                 weight : list = [1,1], angleThreshold : float = 3.14159):
        
        nn.Module.__init__(self)
        self._numPatches = numPatches
        self._kNeighbors = kNeighbors
        self._useNormals = normals
        self._useSurfVar = surfaceVariances
                
        assert(len(weight) == 2 and "input weight has to contain 2 elements, the first for normal, the second to surfaceVar.")
        self._normalWeight  = weight[0]
        self._surfVarWeight = weight[1]
           
        self._normalLossAbs = normalLossAbs
        self._angleThreshold= angleThreshold

        
    def forward(self, pointCloud : Tensor, gtPoints : Tensor = None, gtNormal : Tensor = None):
        
        kNearestNeighbor = getkNearestNeighbor(pointCloud, self._kNeighbors, gtPoints, gtNormal, self._angleThreshold)
        surfacePropDiff  = []
        
        if self._useNormals:
            normalVecGlobal  = estimateNormal(kNearestNeighbor)
            normalPatchwise  = estimatePatchNormal(pointCloud, self._numPatches, self._kNeighbors, self._angleThreshold)
            
            if self._normalLossAbs: 
                # use the absolute value difference between normal vectors
                normalVectorLoss = (normalPatchwise.abs() - normalVecGlobal.abs()).norm(dim=2).mean()[None]
            else:
                # direction insensitive normal loss
                normalVectorLoss = (1 - (normalPatchwise[:,:,None,:] @ normalVecGlobal[:,:,:,None]).squeeze().pow(2)).mean()[None]
            
            surfacePropDiff.append(normalVectorLoss * self._normalWeight)
        
        if self._useSurfVar:
            SurfVarGlobal    = estimateSurfVariance(kNearestNeighbor)
            SurfVarPatchwise = estimatePatchSurfVar(pointCloud, self._numPatches, self._kNeighbors, self._angleThreshold)
            
            SurfVarianceLoss = (SurfVarPatchwise - SurfVarGlobal).pow(2).mean()[None]
        
            surfacePropDiff.append(SurfVarianceLoss * self._surfVarWeight)  
        
        return surfacePropDiff, normalVecGlobal