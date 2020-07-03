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
    
    The gtNormal is used to constraint the knn with the given angle threshold.

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
        as KNN of the targe point. It is optional but need to be given 
        together with gtPoints. 
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
    
    # if knn rejection is enabled
    if gtPoints is not None and gtNormal is not None: 
        dstnumBatchs = gtPoints.shape[0]
        dstnumPoints = gtPoints.shape[1]
        dstdimension = gtPoints.shape[2]
 
        # get the normal of the nearest groundtruth point of each point
        distanceToGT   = ( points.detach().reshape(srcnumBatchs, srcnumPoints, 1, srcdimension)
                          - gtPoints.detach().reshape(dstnumBatchs, 1, dstnumPoints, dstdimension)
                          ).pow(2).sum(dim = 3).sqrt()
        srcNearestInd  = distanceToGT.argmin(dim = 2)
        srcIdealNormal = gtNormal[:, srcNearestInd, :][torch.arange(srcnumBatchs), torch.arange(srcnumBatchs), :, :]
        
        angleBtnormal  = srcIdealNormal @ srcIdealNormal.permute(0,2,1) 
        invalidKnnPt   = angleBtnormal <= torch.tensor(angleThreshold).cos() 
        distanceMatrix[invalidKnnPt] = _USRINF
        
        # This call could be used to verify that the global knn is 
        # constrainted by the angle threshold 
        # visSpecifiedPoints(points[0].detach().to('cpu'), [torch.where(invalidKnnPt[0,769] == 0)[0]])

    colCloseIndice = distanceMatrix.topk(k, dim = 1, largest = False)[1].permute(0,2,1)
    kNearestPoints = points[:, colCloseIndice, :][torch.arange(srcnumBatchs), torch.arange(srcnumBatchs), :, :]
    
    # This call could be used to verify that patchwise knns are not 
    # constrainted by the angle threshold.
    # visSpecifiedPoints(points[0].detach().to('cpu'), [colCloseIndice[0,69]])
    
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
    eigVector = covarianc.to("cpu").symeig( eigenvectors = True )[1][:,:,:,0]
    
    assert(eigVector.shape == torch.Size([kNearestPoints.shape[0], kNearestPoints.shape[1], 3]))
    
    return eigVector.to(kNearestPoints.device)
    

def estimatePatchNormal(points : Tensor, numPatches : int, numNeighbor : int, 
                        gtPoints : Tensor = None, gtNormal : Tensor = None,
                        angleThreshold : float = 3.14159) -> Tensor:
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
            
            patchPC  = points[batch, patchCnt*patchPoint : (patchCnt + 1)*patchPoint, :][None, :, :]   
            
            # reject knn
            if gtPoints is not None and gtNormal is not None:
                kNearest = getkNearestNeighbor(patchPC, numNeighbor, gtPoints[batch][None, :, :],
                                              gtNormal[batch][None, :, :], angleThreshold)
            # do not reject knn
            else:
                kNearest = getkNearestNeighbor(patchPC, numNeighbor, gtPoints, gtNormal, angleThreshold)
                
            normals  = estimateNormal(kNearest)           
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
    eigValues  = covariance.to("cpu").symeig( eigenvectors = True )[0]
    
    # surface variances
    surfaceVar = eigValues[:,:,0]/eigValues.sum(dim=2)
    
    assert(surfaceVar.shape == kNearestPoints.shape[:2])
    
    return surfaceVar.to(kNearestPoints.device)


def estimatePatchSurfVar(points : Tensor, numPatches : int, numNeighbor : int,
                         gtPoints : Tensor = None, gtNormal : Tensor = None,
                         angleThreshold : float = 3.14159) -> Tensor:
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
            
            # reject knn
            if gtPoints is not None and gtNormal is not None:
                kNearest = getkNearestNeighbor(patchPC, numNeighbor, gtPoints[batch][None, :, :],
                                               gtNormal[batch][None, :, :], angleThreshold)
            # do not reject knn
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
        # Since the pytorch.argsort is not stable, it would permulate equal 
        # elements randomly, which could leads to unexpected results. So, we 
        # made the trick here adding acsending small number to the indices.
        
        # points at the left margin, xxPatch is to distinguish patches and 
        # margins, for possible processing.
        leftPoints = points[batch, leftMargin[batch,:], :]
        leftPatch  = torch.where(leftMargin[batch,:])[0]//patchSize \
                     + 0.1 + torch.linspace(0, 0.01, steps=leftPoints.shape[0]).to(leftMargin.device)
        # points at the top margin
        topPoints  = points[batch, topMargin[batch,:], :]
        topPatch   = torch.where(topMargin[batch,:])[0]//patchSize  \
                     + 0.2 + torch.linspace(0, 0.01, steps=topPoints.shape[0]).to(topPoints.device)
        # points at the right margin
        rightPoints = points[batch, rightMargin[batch,:], :]
        rightPatch  = torch.where(rightMargin[batch,:])[0]//patchSize \
                     + 0.3 + torch.linspace(0, 0.01, steps=rightPoints.shape[0]).to(rightPoints.device)
        # points at the bottom margin
        bottPoints = points[batch, bottMargin[batch,:], :]
        bottPatch  = torch.where(bottMargin[batch,:])[0]//patchSize \
                     + 0.4 + torch.linspace(0, 0.01, steps=bottPoints.shape[0]).to(bottPoints.device)
        
        boundaryPoints = torch.cat((leftPoints, topPoints, rightPoints, bottPoints), dim = 0)
        boundaryIndice = torch.cat((leftPatch, topPatch, rightPatch, bottPatch), dim = 0)
        
        # sort the points and indices by patches and by margins
        # 0.1 = left, 0.2 = top, 0.3 = right, 0.4 = bottom
        patchOrder = torch.argsort(boundaryIndice)
        boundaryPoints = boundaryPoints[patchOrder, :]
        boundaryIndice = boundaryIndice[patchOrder]
        
        # margin = torch.zeros_like(points[batch,:,0])
        # margin[leftMargin[batch]] = 1
        # visSpecifiedPoints(points[batch].detach().to('cpu'), [torch.where(margin >= 1)[0].to('cpu')])
        
        # compute distances between points in margins
        distanceMatrix = (boundaryPoints[None, :, :] - boundaryPoints[:, None, :]).norm(dim=2)
        
        # remove distances between point from the same patch
        for ind in range(numPatch):
            subMatrix = torch.where((boundaryIndice < ind + 1) * (boundaryIndice > ind))[0]
            if subMatrix.max()-subMatrix.min()+1 < subMatrix.size(0):
                print("Error here!")
            distanceMatrix[subMatrix.min():subMatrix.max()+1, subMatrix.min():subMatrix.max()+1] = _USRINF
        
        # keep the smallest distance 
        smallestDistance = torch.min(distanceMatrix, dim=1)[0]
        
        # vis distance 
        # visDistance = torch.zeros_like(points[batch][:,0])
        # visDistance[leftMargin[batch]+topMargin[batch]+rightMargin[batch]+bottMargin[batch]]=smallestDistance 
        # visDistance = visDistance[:,None].repeat(1,3)
        # visSpecifiedPoints(points[batch].detach().to('cpu'), 
        #                     [torch.where(visDistance[:,0] > 0)[0].to('cpu')],
        #                     [(visDistance[torch.where(visDistance[:,0] > 0)[0], :]/visDistance.max()).to('cpu')],
        #                     showPatches = True)
        
        # keep the distance and indices for future use
        batchStitchingDist.append(smallestDistance)
        batchStitchingIndex.append(boundaryIndice)
        
        # compute stitching loss for each patch
        # for each margin of the patch, compute the average smallest.
        # Ideally, the sum of the 4 averages is patch stitching loss. However,
        # as the uvspace is uniform distribution, leading to empty margin, 
        # here we have to use mean() to avoid errors. 
        for ind in range(numPatch):
            marginLoss = Tensor([smallestDistance[(boundaryIndice < ind + 0.12)*(boundaryIndice >= ind + 0.1)].mean(),
                                 smallestDistance[(boundaryIndice < ind + 0.22)*(boundaryIndice >= ind + 0.2)].mean(),
                                 smallestDistance[(boundaryIndice < ind + 0.32)*(boundaryIndice >= ind + 0.3)].mean(),
                                 smallestDistance[(boundaryIndice < ind + 0.42)*(boundaryIndice >= ind + 0.4)].mean()])
                    
            batchStitchingLoss += marginLoss[~torch.isnan(marginLoss)].mean()
    
    # average the loss over all batches
    batchStitchingLoss /= batchSize
    
    return batchStitchingLoss


def criterionStitchingFullPatch(uvspace: Tensor, points: Tensor, numPatch: int, 
                                marginSize: float = 0.1):
    '''
    It computes the stitching criterion to evaluate the patch stitching 
    quality. Difference from the criterionStitching(), this function use full 
    patches to compute the stitching errors.

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
    
    batchStitchingLoss = torch.tensor([0.]).to(points.device)
    batchStitchingDist = []
    batchStitchingIndex= []
    for batch in range(batchSize):
        # Since the pytorch.argsort is not stable, it would permulate equal 
        # elements randomly, which could leads to unexpected results. So, we 
        # made the trick here adding acsending small number to the indices.
        
        # points at the left margin, xxPatch is to distinguish patches and 
        # margins, for possible processing.
        leftPoints = points[batch, leftMargin[batch,:], :]
        leftPatch  = torch.where(leftMargin[batch,:])[0]//patchSize \
                     + 0.1 + torch.linspace(0, 0.01, steps=leftPoints.shape[0]).to(leftMargin.device)
        # points at the top margin
        topPoints  = points[batch, topMargin[batch,:], :]
        topPatch   = torch.where(topMargin[batch,:])[0]//patchSize  \
                     + 0.2 + torch.linspace(0, 0.01, steps=topPoints.shape[0]).to(topPoints.device)
        # points at the right margin
        rightPoints = points[batch, rightMargin[batch,:], :]
        rightPatch  = torch.where(rightMargin[batch,:])[0]//patchSize \
                     + 0.3 + torch.linspace(0, 0.01, steps=rightPoints.shape[0]).to(rightPoints.device)
        # points at the bottom margin
        bottPoints = points[batch, bottMargin[batch,:], :]
        bottPatch  = torch.where(bottMargin[batch,:])[0]//patchSize \
                     + 0.4 + torch.linspace(0, 0.01, steps=bottPoints.shape[0]).to(bottPoints.device)
        
        boundaryPoints = torch.cat((leftPoints, topPoints, rightPoints, bottPoints), dim = 0)
        boundaryIndice = torch.cat((leftPatch, topPatch, rightPatch, bottPatch), dim = 0)
        
        # sort the points and indices by patches and by margins
        # 0.1 = left, 0.2 = top, 0.3 = right, 0.4 = bottom
        patchOrder = torch.argsort(boundaryIndice)
        boundaryPoints = boundaryPoints[patchOrder, :]
        boundaryIndice = boundaryIndice[patchOrder]
        
        # margin = torch.zeros_like(points[batch,:,0])
        # margin[leftMargin[batch]] = 1
        # visSpecifiedPoints(points[batch].detach().to('cpu'), [torch.where(margin >= 1)[0].to('cpu')])
        
        # compute distances between points in margins
        distanceMatrix = (boundaryPoints[:, None, :] - points[batch][None, :, :]).norm(dim=2)
        
        # remove distances between point from the same patch
        for ind in range(numPatch):
            subMatrix = torch.where((boundaryIndice < ind + 1) * (boundaryIndice > ind))[0]
            if subMatrix.max()-subMatrix.min()+1 < subMatrix.size(0):
                print("Error here!")
            mask = torch.zeros_like(distanceMatrix)
            mask[subMatrix.min():subMatrix.max()+1, ind*int(patchSize):(ind+1)*int(patchSize)] = _USRINF
            distanceMatrix = distanceMatrix + mask
        
        # keep the smallest distance 
        smallestDistance = torch.min(distanceMatrix, dim=1)[0]
        
        # vis distance 
        # tempDistance= torch.zeros_like(smallestDistance)
        # visDistance = torch.zeros_like(points[batch][:,0])
        # tempDistance[patchOrder] = smallestDistance
        # marginPnum  = tempDistance.shape[0]//4
        
        # visDistance[leftMargin[batch]] = tempDistance[:marginPnum]
        # visDistance[topMargin[batch]]  = tempDistance[marginPnum:2*marginPnum]
        # visDistance[rightMargin[batch]]= tempDistance[2*marginPnum:3*marginPnum]
        # visDistance[bottMargin[batch]] = tempDistance[3*marginPnum:]
        # visDistance = visDistance[:,None].repeat(1,3)
        # visSpecifiedPoints(points[batch].detach().to('cpu'), 
        #                     [torch.where(visDistance[:,0] > 0)[0].to('cpu')],
        #                     [(visDistance[torch.where(visDistance[:,0] > 0)[0], :]/visDistance.max()).to('cpu')],
        #                     showPatches = True)
     
        # keep the distance and indices for future use
        batchStitchingDist.append(smallestDistance)
        batchStitchingIndex.append(boundaryIndice)
        
        # compute stitching loss for each patch
        # for each margin of the patch, compute the average smallest.
        # Ideally, the sum of the 4 averages is patch stitching loss. However,
        # as the uvspace is uniform distribution, leading to empty margin, 
        # here we have to use mean() to avoid errors. 
        for ind in range(numPatch):
            marginLoss = torch.cat([smallestDistance[(boundaryIndice < ind + 0.12)*(boundaryIndice >= ind + 0.1)].mean()[None],
                                    smallestDistance[(boundaryIndice < ind + 0.22)*(boundaryIndice >= ind + 0.2)].mean()[None],
                                    smallestDistance[(boundaryIndice < ind + 0.32)*(boundaryIndice >= ind + 0.3)].mean()[None],
                                    smallestDistance[(boundaryIndice < ind + 0.42)*(boundaryIndice >= ind + 0.4)].mean()[None]])
                    
            batchStitchingLoss += marginLoss[~torch.isnan(marginLoss)].mean()
    
    # average the loss over all batches
    batchStitchingLoss /= batchSize
    
    return batchStitchingLoss[0]
        

def normalAngularDifference(gtPoints: Tensor, gtNormals: Tensor, 
                     predPoints: Tensor, globalNormals: Tensor):
    '''
    It computes the difference of the normal vectors.
    
    For each point, it search for the closest gt point and uses the gt normal 
    of the gt point as the ideal normal of the point. Then, it compute the 
    angular difference between the ideal normal vectors and the globally 
    estimated normal vector.

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
    normalAngDiffAvg : 
        The angluar differnce between the gt normal and the global normal.

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
    normalAngDiffVec = (srcIdealNormal[:,:,None,:] @ globalNormals[:,:,:,None]).squeeze().abs().clamp(0,1).acos()
    normalAngDiffAvg = normalAngDiffVec.mean()

    # visualize the normal, and the normal difference
    # visNormalDiff(predPoints[0].to('cpu'), srcIdealNormal[0].to('cpu'), globalNormals[0].to('cpu'), 25)
    # visNormalDiff(predPoints[0].to('cpu'), 
    #               normalAngDiffVec[:,:,None].repeat(1,1,3)[0].to('cpu'), 
    #               globalNormals[0].to('cpu'), 
    #               25)
    
    return normalAngDiffAvg


def analyticalNormalError(gtPoints: Tensor, gtNormals: Tensor, 
                          predPoints: Tensor, predNormals: Tensor):
    '''
    
    Duplicate of normalAngularDifference. They have exactly the same codes.
    
    It computes the difference betwwen the analytically predicted normal 
    vectors and the ground truth vectors.
    
    For each point, it search for the closest gt point and uses the gt normal 
    of the gt point as the ideal normal of the point. Then, it compute the 
    the angular error between the ideal normal vectors and the analytically 
    estimated normal vector.
    
    Parameters
    ----------
    gtPoints : Tensor
        Ground truth points.
    gtNormals : Tensor
        Ground truth normals.
    predPoints : Tensor
        Predicted points.
    predNormals : Tensor
        Analytically estimated normals.

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
    normalDiffVec = (srcIdealNormal[:,:,None,:] @ predNormals[:,:,:,None]).squeeze().abs().clamp(0, 1).acos()
    normalDiffAvg = normalDiffVec.mean()
    
    # visualize the normal, and the normal difference
    # visNormalDiff(predPoints[0].to('cpu'), srcIdealNormal[0].to('cpu'), predNormals[0].to('cpu'), 25)
    # visNormalDiff(predPoints[0].to('cpu'), 
    #               normalDiffVec[:,:,None].repeat(1,1,3)[0].to('cpu'), 
    #               predNormals[0].to('cpu'), 
    #               25)
    
    return normalDiffAvg


def overlap_criterion(gtPoints: Tensor, predPoints: Tensor, threshold: float,
                      numPatches: int):
    '''
    It computes the overlap criterion of the given predicted points. For each 
    ground truth point, it searches all neighbors within a given threshold 
    and count the number of patches these neighbor coming from and average 
    over all ground truth points.
    
    What is worth to be aware is that for a ground truth point, it is possible 
    there is no predicted point in side its sphere, leading to 0 patches. This
    could bring down the overlap criterion. A solution is only average over 
    the gt points which have nearest neighbor.

    Parameters
    ----------
    gtPoints : Tensor
        Ground truth points, [B, N, 3].
    predPoints : Tensor
        Predicted points, [B, N, 3].
    threshold : float
        The threshold to find neighbors.
    numPatches : int
        The number of patches in the predicted point clouds.

    Returnsotjadjdfsfddfesdfdsfsdfsdfsdfsdf
    -------
    The overlap criterion of the give predicted points.

    '''
    numBatchs = predPoints.shape[0]
    numPoints = predPoints.shape[1]
    dimension = predPoints.shape[2]
    patchSize = numPoints/numPatches
    
    assert(gtPoints.shape[0]==numBatchs and gtPoints.shape[1]==numPoints and gtPoints.shape[2] == dimension)
    
    distanceMatrix = ( predPoints.detach().reshape(numBatchs,  1, numPoints, dimension)
                   - gtPoints.detach().reshape(numBatchs, numPoints, 1, dimension)
                   ).pow(2).sum(dim = 3).sqrt()

    bchInd, gtInd, srcInd  = torch.where(distanceMatrix <= threshold)
    patchInd = srcInd//patchSize
    maskMat= torch.zeros([bchInd.max().int().item()+1, 
                          gtInd.max().int().item()+1,
                          patchInd.max().int().item()+1]).to(predPoints.device)
    
    # visSpecifiedPoints(predPoints[0].detach().to('cpu'), [srcInd[(bchInd == 0)*(gtInd == 1600)]])
    # predPoints[0,srcInd[(bchInd == 0)*(gtInd == 1600)]]
    # gtPoints[0,1600]
    
    # mark all appearing patches for each gt point
    maskMat[bchInd, gtInd, patchInd.long()] += 1
    numOverlap = maskMat.sum(dim=2)
    
    # average over all valid points and then over batches.
    overlapCriterion = torch.zeros([1]).to(predPoints.device)
    for ct in range(numBatchs):
        batchOverlap = numOverlap[ct]
        overlapCriterion += batchOverlap[batchOverlap != 0].mean()
    overlapCriterion /= numBatchs
    
    return overlapCriterion
    
    

class surfacePropLoss(nn.Module):
    
    def __init__(self, numPatches : int, kNeighborsGlobal : int, 
                 kNeighborsPatch : int, normals : bool = True,
                 normalLossAbs : bool = True, surfaceVariances : bool = False,
                 weight : list = [1,1], angleThreshold : float = 3.14159,
                 GlobalandPatch: bool = False, 
                 predNormalforPatch : bool = False):
        
        nn.Module.__init__(self)
        self._numPatches = numPatches
        self._kNeighborG = kNeighborsGlobal
        self._kNeighborP = kNeighborsPatch
        self._useNormals = normals
        self._useSurfVar = surfaceVariances
                
        assert(len(weight) == 2 and "input weight has to contain 2 elements, the first for normal, the second to surfaceVar.")
        self._normalWeight  = weight[0]
        self._surfVarWeight = weight[1]
           
        self._normalLossAbs = normalLossAbs
        self._angleThreshold= angleThreshold
        
        self._GlobalandPatch= GlobalandPatch
        self._predNormalforPatch = predNormalforPatch    # use GT normal as patch-wise normal
        
        print("Surface loss is enabled: \n\tnumNeigborG = %d,\tnumNeigborP = %d,\tuseNormals = %i,\tuseSurfVar = %i,\tangleThres = %f,\tGlob&Patch = %i,\tpredNormalforPatch = %i, \t normalWeig = %f,\t surfWeight = %f"
              %(self._kNeighborG, self._kNeighborP, self._useNormals, self._useSurfVar, self._angleThreshold, self._GlobalandPatch, self._predNormalforPatch, self._normalWeight, self._surfVarWeight))
        
        
    def forward(self, pointCloud : Tensor, predNormal: Tensor, gtPoints : Tensor = None, gtNormal : Tensor = None):
        
        kNearestNeighbor = getkNearestNeighbor(pointCloud, self._kNeighborG, gtPoints, gtNormal, self._angleThreshold)
        surfacePropDiff  = []
        
        
        if self._useNormals:
            normalVecGlobal  = estimateNormal(kNearestNeighbor)
            
            if self._predNormalforPatch:
                # use gt normal as patch-wise normal for comparison. Its
                # priority is higer than Global and Patch.
                normalPatchwise = predNormal
            else:
                if self._GlobalandPatch:
                    # reject invalid KNN points for patches
                    normalPatchwise  = estimatePatchNormal(pointCloud, self._numPatches, self._kNeighborP, gtPoints, gtNormal, self._angleThreshold)
                else:
                    # do not reject invalid KNN for patches
                    normalPatchwise  = estimatePatchNormal(pointCloud, self._numPatches, self._kNeighborP)
                
            if self._normalLossAbs: 
                # use the absolute value difference between normal vectors
                normalVectorLoss = (normalPatchwise.abs() - normalVecGlobal.abs()).norm(dim=2).mean()[None]
            else:
                # direction insensitive normal loss
                normalVectorLoss = (1 - (normalPatchwise[:,:,None,:] @ normalVecGlobal[:,:,:,None]).squeeze().pow(2)).mean()[None]
            
            surfacePropDiff.append(normalVectorLoss * self._normalWeight)
        
        
        if self._useSurfVar:
            SurfVarGlobal    = estimateSurfVariance(kNearestNeighbor)
            
            if self._GlobalandPatch:
                # reject invalid KNN points for patches
                SurfVarPatchwise = estimatePatchSurfVar(pointCloud, self._numPatches, self._kNeighborP, gtPoints, gtNormal, self._angleThreshold)
            else:
                # do not reject invalid KNN for patches
                SurfVarPatchwise = estimatePatchSurfVar(pointCloud, self._numPatches, self._kNeighborP)
            
            SurfVarianceLoss = (SurfVarPatchwise - SurfVarGlobal).pow(2).mean()[None]
        
            surfacePropDiff.append(SurfVarianceLoss * self._surfVarWeight)  
        
        return surfacePropDiff, normalVecGlobal
