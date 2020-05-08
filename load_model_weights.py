from model import AtlasNetReimpl
import helpers
import torch

path_conf = 'config.yaml'
path_weights = '/cvlabdata2/home/zdeng/weights/pcae_shapenet/chkpt_plane.tar'

conf = helpers.load_conf(path_conf)
trstate = torch.load(path_weights)

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
