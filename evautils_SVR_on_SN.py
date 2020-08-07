#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:32:40 2020

@author: zhantao
"""
import os
import glob
from os.path import join as pjn
import numpy as np

import torch
from torch.utils.data import DataLoader

import helpers as helpers
from model import AtlasNetReimplEncImg
from data_loader import ShapeNet, DataLoaderDevice
from sampler import FNSamplerRegularGrid


def compareOurs(path_conf: str, path_weight: str):
    '''
    It compute the stitching error and normal difference for the given model 
    with the given configurations.

    Parameters
    ----------
    path_conf : str
        path to the configuration file.
    path_weight : str
        path to the pretrained model.

    Returns
    -------
    stitchCriterion : list
        Stitiching loss.
    normalDifference : list
        normal difference.

    '''
    # load configuration and weight. check gpu state
    conf = helpers.load_conf(path_conf)
    trstate = torch.load(path_weight)
    gpu  = torch.cuda.is_available()
    
    # subfolder to save predicted point clouds
    folder2save = pjn( '/'.join(path_weight.split('/')[:-1]), 'prediction')
    if not os.path.isdir(folder2save):
        os.mkdir(folder2save)
    
    #### ONLY FOR EVALUATION ####
    conf['loss_patch_area']        = True
    conf['show_overlap_criterion'] = True
    conf['overlap_threshold']    = 0.05
    conf['loss_smooth_surfaces'] = True
    conf['loss_patch_stitching'] = False
    conf['alpha_stitching']      = 0.001
    conf['show_analyticalNormalDiff'] = True
    conf['surface_normal']       = True
    conf['surface_varinace']     = True
    conf['knn_Global']           = 20
    conf['knn_Patch']            = 20
    conf['PredNormalforpatchwise'] = False
    #### ONLY FOR EVALUATION ####

    # resume pretrained model
    model = AtlasNetReimplEncImg(
        M=conf['M'], code= conf['code'], num_patches=conf['num_patches'],
        normalize_cw     = conf['normalize_cw'],
        freeze_encoder   = conf['enc_freeze'],
        enc_load_weights = conf['enc_weights'],
        dec_activ_fns    = conf['dec_activ_fns'],
        dec_use_tanh     = conf['dec_use_tanh'],
        dec_batch_norm   = conf['dec_batch_norm'],
        loss_scaled_isometry  = conf['loss_scaled_isometry'],
        loss_patch_areas      = conf['loss_patch_area'],           # zhantao 
        loss_smooth_surfaces  = conf['loss_smooth_surfaces'],      # zhantao
        loss_patch_stitching  = conf['loss_patch_stitching'],      # zhantao
        numNeighborGlobal     = conf['knn_Global'],                # zhantao
        numNeighborPatchwise  = conf['knn_Patch'],                 # zhantao
        alpha_scaled_isometry = conf['alpha_scaled_isometry'],
        alphas_sciso     = conf['alphas_sciso'], 
        alpha_scaled_surfProp = conf['alpha_surfProp'],            # zhantao
        alpha_stitching  = conf['alpha_stitching'],                # zhantao
        useSurfaceNormal   = conf['surface_normal'],               # zhantao
        useSurfaceVariance = conf['surface_varinace'],             # zhantao
        angleThreshold     = conf['angle_threshold']/180*np.pi,    # zhantao
        rejGlobalandPatch  = conf["reject_GlobalandPatch"],        # zhantao
        predNormalasPatchwise = conf['PredNormalforpatchwise'],    # zhantao
        overlap_criterion  = conf['show_overlap_criterion'],       # zhantao 
        overlap_threshold  = conf['overlap_threshold'],            # zhantao 
        enableAnaNormalErr = conf['show_analyticalNormalDiff'],    # zhantao
        marginSize       = conf['margin_size'],                    # zhantao
        gpu=gpu)

    model.load_state_dict(trstate['weights'])
    
    # using regular grid for evaluation
    model.sampler = FNSamplerRegularGrid( (0., 1.), (0., 1.), 
                                         model._num_patches * model._spp, 
                                         model._num_patches, gpu=gpu)
    
    # prepare data set
    ds_va = ShapeNet(
        conf['path_root_imgs'], conf['path_root_pclouds'],
        conf['path_category_file'], class_choice=conf['va_classes'], train=False, 
        test=False, SVR = True, npoints=conf['N'], load_area=True)
    
    dl_va = DataLoaderDevice( DataLoader(
        ds_va, batch_size = conf['batch_size'], shuffle=False, num_workers=2,   # shuffle is turned off
        drop_last=True), gpu=gpu )
    
    # point cloud inference
    stitchCriterion = []
    normalDifference= []
    ConsistencyLoss = []
    overlapCriterion= []
    analyNormalError= []
    chamferDistance = []
    
    for bi, batch in enumerate(dl_va):
        
        model(batch['img'], it=bi)
        losses = model.loss(batch['pcloud'], normals_gt=batch['normals'], areas_gt=batch['area'])
        
        stitchCriterion.append (losses['Err_stitching'].to('cpu'))
        normalDifference.append(losses['normalDiff'].to('cpu'))
        ConsistencyLoss.append (losses['L_surfProp'].detach().to('cpu'))
        overlapCriterion.append(losses['overlapCriterion'].to('cpu'))
        analyNormalError.append(losses['analyticalNormalDiff'].to('cpu'))
        chamferDistance.append (losses['loss_chd'].detach().to('cpu'))
        
        torch.cuda.empty_cache() 
                    
        # torch.save( model.pc_pred.detach().cpu(), pjn(folder2save, 'regularSample{}.pt'.format(bi))) 
    
    criterion  = torch.cat((torch.tensor(stitchCriterion) [:,None], 
                            torch.tensor(normalDifference)[:,None],
                            torch.tensor(ConsistencyLoss) [:,None],
                            torch.tensor(overlapCriterion)[:,None],
                            torch.tensor(analyNormalError)[:,None],
                            torch.tensor(chamferDistance) [:,None]), dim=1).numpy()
    
    # print(criterion)
    
    # save all results for reference
    error_file = open( pjn( folder2save,'regularSampleFull{}_va_errors.txt'.format(bi)), 'w')
    np.savetxt( error_file, 
                criterion, 
                delimiter=',', header = 'stitching_error, normalAngulardiff, consistency_loss, overlapCriterion, analyticalNorrmalAngularDiff, CHD', comments="#")
    error_file.close()
    
    # save the average error
    avgErr_file = open( pjn( folder2save,'regularSampleFull{}_va_avgErrors.txt'.format(bi)), 'w')
    avgError    = criterion.mean(axis = 0)
    avgError[3] = criterion[criterion[:,3] > 0, 3].mean()    # remove invalid cases before averaging
    np.savetxt( avgErr_file, 
                avgError, 
                delimiter=',', header = 'stitching_error, normalAngulardiff, consistency_loss, overlapCriterion, analyticalNorrmalAngularDiff, CHD', comments="#")
    avgErr_file.close()
        
    return stitchCriterion, normalDifference


def inferenceAll(conf_path: str, weightFolder : str):
    '''
    Given a folder where different weights are stored in different subfolders,
    this function loads these weight and conducts pointcloud inference one by
    one.

    Parameters
    ----------
    conf_path : str
        path to the configuration file.
    weightFolder : str
        path to the folder where each subfolders store different weights.
    numepoch : int, optional
        how many epoches would be used, i.e. how many pointcloud to be 
        inferenced. The default is 4.

    Returns
    -------
    None.

    '''
    for folder in sorted(glob.glob( pjn(weightFolder, '*'))):
        
        if len(glob.glob( pjn(folder, '*.tar') )) == 0:
            continue
        
        weightpath =  glob.glob( pjn(folder, '*.tar') )[0]
        # weightpath = './data/chkpt_cellphone_ep235.tar'
        
        if conf_path is None:
            temp_path = glob.glob( pjn(folder, 'config.yaml') )[0]
            # conf_path = './config.yaml'
            print(temp_path)
            compareOurs(temp_path, weightpath)
            
        else:
            compareOurs(conf_path, weightpath)


path_conf = None

inferenceAll(path_conf, '../models/comparison/SVR_SN/plane')
