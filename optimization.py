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
numNeighbor = 5
num_patches = 25
pointCloud  = torch.load('./ShapeReconstructionNet/data/reconstructedShape/cellphone/estimation.pt').to('cpu')
gtPointCloud= torch.load('./ShapeReconstructionNet/data/reconstructedShape/cellphone/gtData.pt').to('cpu')
gtPtNormals = torch.load('./ShapeReconstructionNet/data/reconstructedShape/cellphone/gtNormal.pt').to('cpu')

# optimization property
maxiters = 100
stepsize = 1e-1
weights  = [1, 1, 0] # the weigh for normal, surfaceVar and chd
 

# In[] optimize the point cloud
chdLsList = []
diffLslist= []
device = "cpu"    # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pointCloud= pointCloud.to(device)
gtPtCloud = gtPointCloud.to(device)
gtNormals = gtPtNormals.to(device)

# start optimization 
y = pointCloud.clone()
surfProp = surfacePropLoss(num_patches, numNeighbor, normals = True, normalLossAbs = False, 
                           surfaceVariances = False , weight  = weights[:2], angleThreshold = 1.0)

for cnt in range(maxiters + 1):

    x = Variable(y, requires_grad = True)

    surfDiff = surfProp.forward(x, gtPointCloud, gtNormals)
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
        
        kNearestNeighbor = getkNearestNeighbor(optimizedPc[batch,:,:][None, :, :], numNeighbor, None, None, 0)
        normalVecGlobal  = estimateNormal(kNearestNeighbor)
        
        visNormalDiff(optimizedPc[batch,:,:], normalVecGlobal[0], normalPatchwise[0], num_patches) 
        visNormalDiff(gtPtCloud[batch], gtNormals[0], gtNormals[0], num_patches) 
        
    if surfProp._useSurfVar:
        
        surfVarPtch = estimatePatchSurfVar(optimizedPc[batch,:,:][None, :, :], num_patches, numNeighbor)
        surfVarPtch = surfVarPtch.repeat(3,1).t()
        
        kNearestNeighbor = getkNearestNeighbor(optimizedPc[batch,:,:][None, :, :], numNeighbor,  None, None, 0)
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