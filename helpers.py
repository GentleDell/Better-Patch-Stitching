""" Helpers classes and functions.

Author: Jan Bednarik, jan.bednarik@epfl.ch
Date: 17.2.2020
"""

# 3rd party.
import torch
import numpy as np
import matplotlib.colors as mcolors

# Python std.
import yaml
import os
import re
from itertools import cycle


class Device:
    """ Creates the computation device.

    Args:
        gpu (bool): Whether to use gpu.
    """
    def __init__(self, gpu=True):
        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            if gpu:
                print('[WARNING] cuda not available, using CPU.')


class TrainStateSaver:
    """ Saves the training state (weights, optimizer and lr scheduler params)
    to file.

    Args:
        path_file (str): Path to file.
        model (torch.nn.Module): Model from which weights are extracted.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim._LRScheduler): LR scheduler.
        verbose (bool): Whether to print debug info.
    """
    def __init__(self, path_file, model=None, optimizer=None, scheduler=None,
                 verbose=False):
        self._path_file = path_file
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._verbose = verbose

        if not os.path.exists(os.path.dirname(path_file)):
            raise Exception('Path "{}" does not exist.'.format(path_file))

        for var, name in zip([model, optimizer, scheduler],
                             ['model', 'optimizer', 'scheduler']):
            if var is None:
                print('[WARNING] TrainStateSaver: {} is None and will not be '
                      'saved'.format(name))

    def get_file_path(self):
        return self._path_file

    def __call__(self, file_path_override=None, **kwargs):
        state = kwargs
        if self._model:
            state['weights'] = self._model.state_dict()
        if self._optimizer:
            state['optimizer'] = self._optimizer.state_dict()
        if self._scheduler:
            state['scheduler'] = self._scheduler.state_dict()

        # Get the output file path and save.
        path_file = (file_path_override,
                     self._path_file)[file_path_override is None]
        pth_tmp = path_file

        # Try to save the file.
        try:
            torch.save(state, pth_tmp)
        except Exception as e:
            print('ERROR: The model weights file {} could not be saved and '
                  'saving is skipped. The exception: "{}"'.
                  format(pth_tmp, e))
            if os.path.exists(pth_tmp):
                os.remove(pth_tmp)
            return

        if self._verbose:
            print('[INFO] Saved training state to {}'.format(path_file))


class RunningLoss:
    def __init__(self):
        self.reset()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            it = self._its.get(k, 0)
            self._data[k] = self._data.get(k, 0) * (it/(it + 1)) + v/(it + 1)
            self._its[k] = it + 1

    def reset(self):
        self._data = {}
        self._its = {}

    def get_losses(self):
        return self._data.copy()


def identity(x):
    """ Implements the identity function.
    """
    return x


def load_conf(path):
    """ Returns the loaded .cfg config file.

    Args:
        path (str): Aboslute path to .cfg file.

    Returns:
    dict: Loaded config file.
    """

    with open(path, 'r') as f:
        conf = yaml.full_load(f)
    return conf


def jn(*parts):
    """ Returns the file system path composed of `parts`.

    Args:
        *parts (str): Path parts.

    Returns:
        str: Full path.
    """
    return os.path.join(*parts)


def ls(path, exts=None, pattern_incl=None, pattern_excl=None,
       ignore_dot_underscore=True):
    """ Lists the directory and returns it sorted. Only the files with
    extensions in `ext` are kept. The output should match the output of Linux
    command "ls". It wrapps os.listdir() which is not guaranteed to produce
    alphanumerically sorted items.

    Args:
        path (str): Absolute or relative path to list.
        exts (str or list of str or None): Extension(s). If None, files with
            any extension are listed. Each e within `exts` can (but does
            not have to) start with a '.' character. E.g. both
            '.tiff' and 'tiff' are allowed.
        pattern_incl (str): regexp pattern, if not found in the file name,
            the file is not listed.
        pattern_excl (str): regexp pattern, if found in the file name,
            the file is not listed.
        ignore_dot_underscore (bool): Whether to ignore files starting with
            '._' (usually spurious files appearing after manipulating the
            linux file system using sshfs)

    Returns:
        list of str: Alphanumerically sorted list of files contained in
        directory `path` and having extension `ext`.
    """
    if isinstance(exts, str):
        exts = [exts]

    files = [f for f in sorted(os.listdir(path))]

    if exts is not None:
        # Include patterns.
        extsstr = ''
        for e in exts:
            extsstr += ('.', '')[e.startswith('.')] + '{}|'.format(e)
        patt_ext = '({})$'.format(extsstr[:-1])
        re_ext = re.compile(patt_ext)
        files = [f for f in files if re_ext.search(f)]

    if ignore_dot_underscore:
        re_du = re.compile('^\._')
        files = [f for f in files if not re_du.match(f)]

    if pattern_incl is not None:
        re_incl = re.compile(pattern_incl)
        files = [f for f in files if re_incl.search(f)]

    if pattern_excl is not None:
        re_excl = re.compile(pattern_excl)
        files = [f for f in files if not re_excl.search(f)]

    return files


def lsd(path, pattern_incl=None, pattern_excl=None):
    """ Lists directories within path.

    Args:
        path (str): Absolue path to containing dir.
        pattern_incl (str): regexp pattern, if not found in the dir name,
            the dir is not listed.
        pattern_excl (str): regexp pattern, if found in the dir name,
            the dir is not listed.

    Returns:
        list: Directories within `path`.
    """
    if pattern_incl is None and pattern_excl is None:
        dirs = [d for d in ls(path) if os.path.isdir(jn(path, d))]
    else:
        if pattern_incl is None:
            pattern_incl = '^.'
        if pattern_excl is None:
            pattern_excl = '^/'

        pincl = re.compile(pattern_incl)
        pexcl = re.compile(pattern_excl)
        dirs = [d for d in ls(path) if
                os.path.isdir(jn(path, d)) and
                pincl.search(d) is not None and
                pexcl.search(d) is None]

    return dirs


def load_obj(pth):
    """ Loads mesh from .obj file.

    Args:
        pth (str): Absolute path.

    Returns:
        np.array (float): Mesh, (V, 3).
        np.array (int32): Triangulation, (T, 3).
    """
    with open(pth, 'r') as f:
        lines = f.readlines()

    mesh = []
    tri = []
    for l in lines:
        # vals = l.split(' ')
        vals = l.split()
        if len(vals) > 0:
            if vals[0] == 'v':
                mesh.append([float(n) for n in vals[1:]])
            elif vals[0] == 'f':
                tri.append([int(n.split('/')[0]) - 1 for n in vals[1:]])

    mesh = np.array(mesh, dtype=np.float32)
    tri = np.array(tri, dtype=np.int32)

    return mesh, tri


def mesh_area(verts, faces):
    """ Computes the area of the surface as a sum of areas of all the triangles.

    Args:
        verts (np.array of float): Vertices, shape (V, 3).
        faces (np.array of int32): Faces, shape (F, 3)

    Returns:
        float: Total area [m^2].
    """
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3

    v1s, v2s, v3s = verts[faces.flatten()].\
        reshape((-1, 3, 3)).transpose((1, 0, 2))  # each (F, 3)
    v21s = v2s - v1s
    v31s = v3s - v1s
    return np.sum(0.5 * np.linalg.norm(np.cross(v21s, v31s), axis=1))


def get_contrast_colors():
    """ Returns 67 contrast colors.

    Returns:
        dict (str -> tuple): Colors, name -> (R, G, B), values in [0, 1].
    """
    clr_names = \
        ['aqua', 'blue', 'brown', 'chartreuse', 'chocolate', 'coral',
         'cornflowerblue', 'crimson', 'darkblue', 'darkcyan', 'darkgoldenrod',
         'darkgreen', 'darkmagenta', 'darkolivegreen', 'darkorange',
         'darkorchid', 'darkred', 'darkslateblue', 'darkturquoise',
         'darkviolet', 'deeppink', 'deepskyblue', 'firebrick', 'forestgreen',
         'gold', 'goldenrod', 'green', 'greenyellow', 'hotpink', 'indianred',
         'indigo', 'lawngreen', 'lightsalmon', 'lightseagreen', 'lime',
         'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue',
         'mediumorchid', 'mediumseagreen', 'mediumslateblue',
         'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
         'midnightblue', 'navy', 'olive', 'orange', 'orangered', 'peru',
         'purple', 'rebeccapurple', 'red', 'royalblue', 'saddlebrown', 'salmon',
         'seagreen', 'sienna', 'slateblue', 'springgreen', 'steelblue', 'teal',
         'tomato', 'yellow', 'yellowgreen']
    return {k: mcolors.to_rgb(mcolors.CSS4_COLORS[k]) for k in clr_names}


def get_patches_colors(mode, conf=None, M=None):
    """ Returns the per-point color.

    Args:
        mode (str): One of:
            'same': Constant green color.
            'uv': Foldingnet style UV space color encoding.
            'patches': Each patch has different unifomr color.
        conf (dict): Config.
        M (int): # points.

    Returns:
        np.array of float32: Per-point color, shape (N, 3).
    """

    if mode == 'same':
        clrs = 'green'
    elif mode == 'patches':
        n_patches = conf['num_patches']
        spp = M // n_patches
        assert(np.isclose(spp, M / n_patches))
        clrs_cycle = cycle(list(get_contrast_colors().values()))
        clrs = np.ones((M, 3), dtype=np.float32)
        for i in range(n_patches):
            clr = np.array(next(clrs_cycle))
            clrs[i * spp:(i + 1) * spp] *= clr[None]
    else:
        raise Exception('Unsupported mode "{}"'.format(mode))

    return clrs


def pclouds2vis(pcgt, pcp, num_disp, conf):
    """ Converts the GT and predicted pclouds to the format suitable for
    Tensorboard visualization - For every sample the GT and predicted pclouds
    are visualized separately, GT is gray, predicted is colored by patches.

    Args:
        pcgt (torch.Tensor): GT pcloud, shape (B, N, 3)
        pcp (torch.Tensor): Pred. pcloud, shape (B, M, 3)
        num_disp (int): Number of displayed pclouds.
        conf (dict): Model config. file.

    Returns:
        pcs (torch.Tensor of float32): Pclouds to visualize (num_disp, 2, P, 3),
            P is max(M, N).
        clrs (torch.Tensor of uint8): Per-point colors (num_disp, 2, P, 3)
    """
    B, N = pcgt.shape[:2]
    M = pcp.shape[1]
    assert(pcp.shape[0] == B)
    P = np.maximum(N, M)

    pcgt = torch.cat([pcgt, torch.zeros(
        (B, P - N, 3), dtype=torch.float32)], dim=1)  # (B, P, 3)
    pcp = torch.cat([pcp, torch.zeros(
        (B, P - M, 3), dtype=torch.float32)], dim=1)  # (B, P, 3)
    assert pcgt.shape == (B, P, 3)
    assert pcp.shape == (B, P, 3)

    clrs_gt = torch.ones((B, P, 3), dtype=torch.uint8) * 127  # (B, P, 3)
    clrs_pred = torch.from_numpy(np.tile(
        (get_patches_colors('patches', conf=conf, M=M) * 255.0).
            astype(np.uint8), (B, 1, 1)))  # (B, M, 3)
    clrs_pred = torch.cat([clrs_pred, torch.zeros(
        (B, P - M, 3), dtype=torch.uint8)], dim=1)  # (B, P, 3)
    assert clrs_gt.shape == pcgt.shape
    assert clrs_pred.shape == pcp.shape
    assert clrs_gt.dtype == torch.uint8
    assert clrs_pred.dtype == torch.uint8

    pcs = torch.cat([pcgt[:, None], pcp[:, None]], dim=1)[:num_disp] * \
          torch.tensor([1., -1., -1.])
    clrs = torch.cat([clrs_gt[:, None], clrs_pred[:, None]], dim=1)[:num_disp]

    return pcs, clrs
