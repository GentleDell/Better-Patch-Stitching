""" Training of AtlasNet [1] with differential geometry properties based
    regularizers [2] for point-cloud auto-encoding task on ShapeNet.

[1] Groueix Thibault et al. AtlasNet: A Papier-Mâché Approach to Learning 3D
    Surface Generation. CVPR 2018.
[2] Bednarik Jan et al. Shape Reconstruction by Learning Differentiable Surface
    Representations. CoRR 2019.

Author: Jan Bednarik, jan.bednarik@epfl.ch
Date: 17.2.2020
"""

# Python std
import argparse
from timeit import default_timer as timer

# project files
import helpers
from model import AtlasNetReimpl
from data_loader import ShapeNet, DataLoaderDevice

# 3rd party
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

# Settings.
print_loss_tr_every = 50
save_collapsed_every = 50
gpu = torch.cuda.is_available()

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='Path to the main config file of the model.',
                    default='config.yaml')
parser.add_argument('--output', help='Path to the output directory for storing '
                                     'weights and tensorboard data.',
                    default='./data')
parser.add_argument('--resume', help ='Resume training from the given path', 
                    default = False)
args = parser.parse_args()

# Load the config file, prepare paths.
conf = helpers.load_conf(args.conf)

# Model type, color mode.
model_type = 'atlasnet_orig'

# Prepare TB writers.
writer_tr = SummaryWriter(helpers.jn(args.output, 'tr'))
writer_va = SummaryWriter(helpers.jn(args.output, 'va'))

# Build a model.
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
    enableAnaNormalErr = conf['show_analyticalNormalDiff'],    # zhantao
    marginSize       = conf['margin_size'],                    # zhantao
    gpu=gpu)

# Create data loaders.
ds_tr = ShapeNet(
    conf['path_root_imgs'], conf['path_root_pclouds'],
    conf['path_category_file'], class_choice=conf['tr_classes'], train=True,
    npoints=conf['N'], load_area=True)
ds_va = ShapeNet(
    conf['path_root_imgs'], conf['path_root_pclouds'],
    conf['path_category_file'], class_choice=conf['va_classes'], train=False,
    npoints=conf['N'], load_area=True)
dl_tr = DataLoaderDevice(DataLoader(
    ds_tr, batch_size=conf['batch_size'], shuffle=True, num_workers=4,
    drop_last=True), gpu=gpu)
dl_va = DataLoaderDevice(DataLoader(
    ds_va, batch_size=conf['batch_size'], shuffle=True, num_workers=2,
    drop_last=True), gpu=gpu)

print('Train ds: {} samples'.format(len(ds_tr)))
print('Valid ds: {} samples'.format(len(ds_va)))

# Prepare training.
opt = torch.optim.Adam(model.parameters(), lr=conf['lr'])

# Resume training if required
if args.resume:
    print("Resuming training")
    trstate = torch.load(helpers.jn(args.output, 'chkpt.tar'))
    model.load_state_dict(trstate['weights'])
    opt.load_state_dict(trstate['optimizer'])

scheduler = None
if conf['reduce_lr_on_plateau']:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=conf['lr_factor'], patience=conf['lr_patience'],
        min_lr=conf['lr_min'], threshold=conf['lr_threshold'], verbose=True)

# Prepare savers.
saver = helpers.TrainStateSaver(
    helpers.jn(args.output, 'chkpt.tar'), model=model, optimizer=opt,
    scheduler=scheduler)

# Training loop.
iters_tr = int(np.ceil(len(ds_tr) / float(conf['batch_size'])))
iters_va = int(np.ceil(len(ds_va) / float(conf['batch_size'])))
losses_tr = helpers.RunningLoss()
losses_va = helpers.RunningLoss()
for ep in range(1, conf['epochs'] + 1):
    # Training.
    tstart = timer()
    model.train()
    for bi, batch in enumerate(dl_tr, start=1):
        it = (ep - 1) * iters_tr + bi
        model(batch['pcloud'], it=it)
        losses = model.loss(batch['pcloud'], normals_gt=batch['normals'], areas_gt=batch['area'])

        opt.zero_grad()
        losses['loss_tot'].backward()
        opt.step()

        losses_tr.update(**{k: v.item() for k, v in losses.items()})
        if bi % print_loss_tr_every == 0:
            losses_avg = losses_tr.get_losses()
            for k, v in losses_avg.items():
                writer_tr.add_scalar(k, v, it)
            losses_tr.reset()
            writer_tr.add_scalar('lr', opt.param_groups[0]['lr'], it)

            strh = '\rep {}/{}, it {}/{}, {:.0f} s - '.\
                format(ep, conf['epochs'], bi, iters_tr, timer() - tstart)
            strl = ', '.join(['{}: {:.6f}'.format(k, v)
                              for k, v in losses_avg.items()])
            print(strh + strl, end='')

        # Save pclouds.
        if bi == 1 and conf['pcloud_save_period'] != 0 and \
                ep % conf['pcloud_save_period'] == 0:
            pcs_vis, clrs_vis = helpers.pclouds2vis(
                batch['pcloud'].cpu(), model.pc_pred.detach().cpu(),
                conf['pcloud_samples_per_period'], conf)
            assert pcs_vis.shape == (
                np.minimum(conf['pcloud_samples_per_period'],
                           conf['batch_size']), 2,
                np.maximum(conf['N'], conf['M']), 3)
            assert clrs_vis.shape == pcs_vis.shape
            for idx, (pc, clr) in enumerate(zip(pcs_vis, clrs_vis)):  # (2, P,3)
                writer_tr.add_mesh('pc_{}'.format(idx), vertices=pc, colors=clr,
                                   global_step=it)

        # Save number of collapsed patches.
        if bi % save_collapsed_every == 0 and 'fff' in model.geom_props:
            num_collpased = np.sum(
                [inds.shape[0] for inds in model.collapsed_patches_A()]) / \
                            model.pc_pred.shape[0]
            writer_tr.add_scalar('collapsed_patches', num_collpased,
                                 global_step=it)

    # Validation.
    model.eval()
    it = ep * iters_tr
    loss_va_run = 0.
    for bi, batch in enumerate(dl_va):
        curr_bs = batch['pcloud'].shape[0]
        model(batch['pcloud'])
        lv = model.loss(batch['pcloud'], normals_gt=batch['normals'], areas_gt=batch['area'])['loss_tot']
        loss_va_run += lv.item() * curr_bs

        # Save pclouds.
        if bi == 1 and conf['pcloud_save_period'] != 0 and \
                ep % conf['pcloud_save_period'] == 0:
            pcs_vis, clrs_vis = helpers.pclouds2vis(
                batch['pcloud'].cpu(), model.pc_pred.detach().cpu(),
                conf['pcloud_samples_per_period'], conf)
            assert pcs_vis.shape == (
                np.minimum(conf['pcloud_samples_per_period'],
                           conf['batch_size']), 2,
                np.maximum(conf['N'], conf['M']), 3)
            assert clrs_vis.shape == pcs_vis.shape
            for idx, (pc, clr) in enumerate(zip(pcs_vis, clrs_vis)):  # (2, P,3)
                writer_va.add_mesh('pc_{}'.format(idx), vertices=pc, colors=clr,
                                   global_step=it)

        # Save number of collapsed patches.
        if bi == 1 and 'fff' in model.geom_props:
            num_collpased = np.sum(
                [inds.shape[0] for inds in model.collapsed_patches_A()]) / \
                            model.pc_pred.shape[0]
            writer_va.add_scalar('collapsed_patches', num_collpased,
                                 global_step=it)


    loss_va = loss_va_run / len(ds_va)
    writer_va.add_scalar('loss_tot', loss_va, it)
    print(' ltot_va: {:.4f}'.format(loss_va))

    # LR scheduler.
    if conf['reduce_lr_on_plateau']:
        scheduler.step(loss_va)

    # Save train state.
    saver(epoch=ep, iterations=it)
