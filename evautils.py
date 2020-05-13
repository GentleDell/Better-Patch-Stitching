#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:32:40 2020

@author: zhantao
"""

import torch
from torch.utils.data import DataLoader

import helpers as helpers
from model import AtlasNetReimpl
from data_loader import ShapeNet, DataLoaderDevice
from sampler import FNSamplerRegularGrid

# disable gradient globally
torch.set_grad_enabled(False) 


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
        normalize_cw=conf['normalize_cw'],
        freeze_encoder=conf['enc_freeze'],
        enc_load_weights=conf['enc_weights'],
        dec_activ_fns=conf['dec_activ_fns'],
        dec_use_tanh=conf['dec_use_tanh'],
        dec_batch_norm=conf['dec_batch_norm'],
        loss_scaled_isometry=conf['loss_scaled_isometry'],
        alpha_scaled_isometry=conf['alpha_scaled_isometry'],
        alphas_sciso=conf['alphas_sciso'], gpu=True)
    model.load_state_dict(trstate['weights'])
    
    # using regular grid for evaluation
    model.sampler = FNSamplerRegularGrid( (0., 1.), (0., 1.), 
                                         model._num_patches * model._spp, 
                                         model._num_patches, gpu=gpu)
    
    # prepare data set
    ds_va = ShapeNet(
        conf['path_root_imgs'], conf['path_root_pclouds'],
        conf['path_category_file'], class_choice=conf['va_classes'], train=False,
        npoints=conf['N'], load_area=True)
    
    dl_va = DataLoaderDevice( DataLoader(
        ds_va, batch_size = conf['batch_size'], shuffle=False, num_workers=2,   # shuffle is turned off
        drop_last=True), gpu=gpu )
    
    # point cloud inference
    stitchCriterion = []
    normalDifference= []
    for bi, batch in enumerate(dl_va):
        
        model(batch['pcloud'])
        losses = model.loss(batch['pcloud'], normals_gt=batch['normals'], areas_gt=batch['area']).detach()
        
        stitchCriterion.append(losses['Err_stitching'])
        normalDifference.append(losses[ 'normalDiff' ])
        
    return stitchCriterion, normalDifference