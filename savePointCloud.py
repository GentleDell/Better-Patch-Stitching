#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:56:03 2020

@author: zhantao
"""
import glob
import argparse
from os.path import join as pjn

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import ShapeNet, DataLoaderDevice
from model import AtlasNetReimpl
import helpers as helpers


def pcInference(path_conf: str, path_weight: str, path_save: str, numepoch: int = 1):
    '''
    It reconstructs and saves point clouds, given the path to the configuration
    file, pretrained weight and the path to save the inferenced point cloud. 
    
    This function has to be run under the training folder, i.e. where the 
    model.py and helpers.py are.

    Parameters
    ----------
    path_conf : str
        path to the configuration file for training.
    path_weight : str
        path to the pretrained weight.
    path_save : str
        where to save the inferenced point cloud.
    numepoch : int
        the number of batches to do point cloud inference.

    Returns
    -------
    None.

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
    
    # prepare data set
    ds_va = ShapeNet(
        conf['path_root_imgs'], conf['path_root_pclouds'],
        conf['path_category_file'], class_choice=conf['va_classes'], train=False,
        npoints=conf['N'], load_area=True)
    
    dl_va = DataLoaderDevice( DataLoader(
        ds_va, batch_size = conf['batch_size'], shuffle=False, num_workers=2,   # shuffle is turned off
        drop_last=True), gpu=gpu )
        
    # point cloud inference
    for e in range(numepoch):
        for bi, batch in enumerate(dl_va):
            model(batch['pcloud'])
            torch.save( model.pc_pred.detach().cpu(), pjn( path_save, 'pc{}.pt'.format(bi + e*conf['batch_size']) ) )
            torch.save( batch['pcloud'].cpu(), pjn( path_save, 'gtpc{}.pt'.format(bi + e*conf['batch_size']) ) )

def inferenceAll(conf_path: str, weightFolder : str, save_path : str = None, 
                 numepoch: int = 4):
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
    save_path : str, optional
        where to save the inferenced pointcloud. The default is None meaning 
        that store the pointcloud to the same folder of the weight.
    numepoch : int, optional
        how many epoches would be used, i.e. how many pointcloud to be 
        inferenced. The default is 4.

    Returns
    -------
    None.

    '''
    for folder in sorted(glob.glob( pjn(weightFolder, '*'))):
        weightpath =  glob.glob( pjn(folder, '*.tar') )[0]
        
        if save_path == None:
            savePath = '/'.join(weightpath.split('/')[:-1])
            
        pcInference(conf_path, weightpath, savePath, numepoch)
        
        
path_conf = 'config.yaml'
# path_weights = '../../syn_server/data/cellphone/cellphone_epoch717_0.001_k5_normal/chkpt_cellphone_ep717_weight0.001_k5_normal.tar'
# path_save = '../../syn_server/data/cellphone/cellphone_epoch717_0.001_k5_normal'

inferenceAll(path_conf, '../../syn_server/data/cellphone/newKnn_gtnormal/', numepoch = 1)