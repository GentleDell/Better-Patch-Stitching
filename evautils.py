#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:32:40 2020

@author: zhantao
"""
import glob
from os.path import join as pjn
import numpy as np

import torch
from torch.utils.data import DataLoader

import helpers as helpers
from model import AtlasNetReimpl
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

    # resume pretrained model
    model = AtlasNetReimpl(
        M=conf['M'], code=conf['code'], num_patches=conf['num_patches'],
        normalize_cw     = conf['normalize_cw'],
        freeze_encoder   = conf['enc_freeze'],
        enc_load_weights = conf['enc_weights'],
        dec_activ_fns    = conf['dec_activ_fns'],
        dec_use_tanh     = conf['dec_use_tanh'],
        dec_batch_norm   = conf['dec_batch_norm'],
        loss_scaled_isometry  = conf['loss_scaled_isometry'],
        loss_smooth_surfaces  = conf['loss_smooth_surfaces'],      # zhantao
        loss_patch_stitching  = conf['loss_patch_stitching'],      # zhantao
        numNeighbor      = conf['number_k_neighbor'],              # zhantao
        alpha_scaled_isometry = conf['alpha_scaled_isometry'],
        alphas_sciso     = conf['alphas_sciso'], 
        alpha_scaled_surfProp = conf['alpha_surfProp'],            # zhantao
        alpha_stitching  = conf['alpha_stitching'],                # zhantao
        useSurfaceNormal   = conf['surface_normal'],               # zhantao
        useSurfaceVariance = conf['surface_varinace'],             # zhantao
        angleThreshold     = conf['angle_threshold']/180*np.pi,    # zhantao
        rejGlobalandPatch  = conf["reject_GlobalandPatch"],        # zhantao
        rejByPredictNormal = conf['reject_byPredNormal'],          # zhantao
        overlap_criterion  = conf['show_overlap_criterion'],       # zhantao 
        overlap_threshold  = conf['overlap_threshold'],            # zhantao 
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
        conf['path_category_file'], class_choice=conf['va_classes'], train=True,  # to use 80% of the data 
        npoints=conf['N'], load_area=True)
    
    dl_va = DataLoaderDevice( DataLoader(
        ds_va, batch_size = conf['batch_size'], shuffle=False, num_workers=2,   # shuffle is turned off
        drop_last=True), gpu=gpu )
    
    # point cloud inference
    stitchCriterion = []
    normalDifference= []
    ConsistencyLoss = []
    overlapCriterion= []
    for bi, batch in enumerate(dl_va):
        
        model(batch['pcloud'])
        losses = model.loss(batch['pcloud'], normals_gt=batch['normals'], areas_gt=batch['area'])
        
        stitchCriterion.append(losses['Err_stitching'].to('cpu'))
        normalDifference.append(losses[ 'normalDiff' ].to('cpu'))
        ConsistencyLoss.append(losses['L_surfProp'].to('cpu'))
        overlapCriterion.appen(losses['overlapCriterion'].to('cpu'))
        
        # torch.save( model.pc_pred.detach().cpu(), pjn( '/'.join(path_weight.split('/')[:-1]), 'regularSample{}.pt'.format(bi))) 
    
    criterion  = torch.cat((torch.tensor(stitchCriterion) [:,None], 
                            torch.tensor(normalDifference)[:,None],
                            torch.tensor(ConsistencyLoss) [:,None],
                            torch.tensor(overlapCriterion)[:,None]), dim=1).numpy()
    
    error_file = open( pjn( '/'.join(path_weight.split('/')[:-1]),'regularSampleFull{}_errors.txt'.format(bi)), 'w')
    np.savetxt( error_file, 
                criterion, 
                delimiter=',', header = 'stitching_error, normal_diff, consistency_loss, overlapCriterion', comments="#")
    error_file.close()
    
    avgErr_file = open( pjn( '/'.join(path_weight.split('/')[:-1]),'regularSampleFull{}_avgErrors.txt'.format(bi)), 'w')
    np.savetxt( avgErr_file, 
                criterion.mean(axis = 0), 
                delimiter=',', header = 'stitching_error, normal_diff, consistency_loss, overlapCriterion', comments="#")
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
        weightpath =  glob.glob( pjn(folder, '*.tar') )[0]
        
        if conf_path is None:
            conf_path = glob.glob( pjn(folder, 'config.yaml') )[0]
            print(conf_path)
            
        compareOurs(conf_path, weightpath)


path_conf = None

inferenceAll(path_conf, '../../syn_server/data/cellphone/Knn_gtnormal_bothGlobalandPatchwise/')
