#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:50:25 2020

@author: zhantao
"""

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from visutils import visNormalDiff
from estimateSurfaceProps import surfacePropLoss, chamferDistance, getkNearestNeighbor, \
                                 estimateNormal, estimatePatchNormal, estimateSurfVariance, estimatePatchSurfVar

# patch property
numNeighbor = 8
num_patches = 25
pointCloud  = torch.cat([torch.load('./ShapeReconstructionNet/data/reconstructedShape/pc1.pt')[0][None, :],
                         torch.load('./ShapeReconstructionNet/data/reconstructedShape/pc0.pt')[0][None, :]], dim = 0)

# optimization property
maxiters = 100
stepsize = 1e-1
weights  = [1, 1, 1] # the weigh for normal, surfaceVar and chd
 

# In[] optimize the point cloud
chdLsList = []
diffLslist= []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pointCloud= pointCloud.to(device)

# start optimization 
y = pointCloud.clone()
surfProp = surfacePropLoss(num_patches, numNeighbor, normals = True, normalLossAbs = False, 
                           surfaceVariances = True , weight  = weights[:2])

for cnt in range(maxiters + 1):

    x = Variable(y, requires_grad = True)

    surfDiff = surfProp.forward(x)
    surfLoss = torch.cat(surfDiff).sum()
    
    chdLoss  = chamferDistance(x, pointCloud)
    
    loss = surfLoss + weights[-1]*chdLoss    
    loss.backward()    

    with torch.no_grad():
        y = x - stepsize*x.grad
    
    print("---> iter %d \n\t diffLoss = %.6f, \t chdLoss = %.6f"%( cnt, surfLoss.detach(), chdLoss.detach() ) )
    chdLsList.append(chdLoss.detach())
    diffLslist.append(surfLoss.detach())
    

# In[]
# visualize optimized points and their surface properties
    
optimizedPc = y
if len(optimizedPc.shape) == 2:
    optimizedPc = optimizedPc[None,:,:]

for batch in range(optimizedPc.shape[0]):
    
    if surfProp._useNormals:
        normalPatchwise  = estimatePatchNormal(optimizedPc[batch,:,:][None, :, :], num_patches, numNeighbor)
        
        kNearestNeighbor = getkNearestNeighbor(optimizedPc[batch,:,:][None, :, :], numNeighbor)
        normalVecGlobal  = estimateNormal(kNearestNeighbor)
        
        visNormalDiff(optimizedPc[batch,:,:], normalVecGlobal[0], normalPatchwise[0], num_patches) 
        
    if surfProp._useSurfVar:
        
        surfVarPtch = estimatePatchSurfVar(optimizedPc[batch,:,:][None, :, :], num_patches, numNeighbor)
        surfVarPtch = surfVarPtch.repeat(3,1).t()
        
        kNearestNeighbor = getkNearestNeighbor(optimizedPc[batch,:,:][None, :, :], numNeighbor)
        surfaceVarGlobal = estimateSurfVariance(kNearestNeighbor)
        surfaceVarGlobal = surfaceVarGlobal.repeat(3,1).t()
        
        maxAlign = max(surfVarPtch.max(), surfaceVarGlobal.max())
        surfVarPtch = surfVarPtch/maxAlign
        surfaceVarGlobal = surfaceVarGlobal/maxAlign
        
        visNormalDiff(optimizedPc[batch,:,:], surfaceVarGlobal, surfVarPtch, num_patches)


plt.figure(figsize = [12,6])
plt.subplot(121)
plt.plot(diffLslist)
plt.grid()
plt.xlabel('num of iterations', fontsize = 15)
plt.ylabel("Normal Loss", fontsize = 15)

plt.subplot(122)
plt.plot(chdLsList)
plt.grid()
plt.xlabel('num of iterations', fontsize = 15)
plt.ylabel("chamfer distance", fontsize = 15)

plt.tight_layout()