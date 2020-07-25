# 3rd party
from torch.utils.data import Dataset
import numpy as np
import torch

# project files
import jblib.file_sys as jbfs
import jblib.img as jbim
import jblib.depth as jbd
import jblib.mesh as jbm
import helpers


class DataToTensor:
    """ Converts the numpy arrays to torch Tensors, storage is shared.
    """
    def __call__(self, sample):
        """
        Args:
            sample (dict): Dict containing np.array as values.

        Returns:
            dict: Dict containing torch.Tensor as values.
        """
        sample = sample.copy()
        for k, v in sample.items():
            sample[k] = torch.from_numpy(v)
        return sample


class ImgToTensor:
    """ Converts the np.array tensors of shape (H, W, C), to torch.Tensor
    of shape (C, H, W). (H, W) is spatial dimension, C is # channels.
    """
    def __init__(self, key='img'):
        self._df = key

    def __call__(self, sample):
        """
        Args:
            sample (dict): Dict containing np.array as value at key 'img'.

        Returns:
            dict: Key 'img' mapping to torch.Tensor.
        """
        sample = sample.copy()

        # If grayscale 2D, add a dimension for 1 channel.
        if sample[self._df].ndim == 2:
            sample[self._df] = sample[self._df][..., None]

        sample[self._df] = \
            torch.from_numpy(sample[self._df].transpose((2, 0, 1)))
        return sample


class ImgDataset(Dataset):
    """ Loads RGB images. Hardcoded paths are optimized for the dataset
    Textureless Deformable Surfaces [1] dataset.

    [1] https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/

    Args:
        path_root (str): Path Root path to the ds.
        objects (list of str): Names of dirs containing data for objects.
        obj_seqs (dict): Names of dirs containing object data and required
            sequences. E.g.:
            obj_seqs = {'cloth_square': ['L0_left_edge', 'L1_top_edge'],
                        'tshirt': ['L1_back']}
        transform (callable or str): Tf to be applied on the image data.
    """
    path_infix = 'images'

    def __init__(self, path_root, obj_seqs, exts=('tiff', ),
                 transform='default'):
        super(ImgDataset, self).__init__()
        self._path_root = path_root

        if isinstance(transform, str):
            if transform == 'default':
                transform = ImgToTensor()
            else:
                raise Exception('Unknown transform option "{}"'.
                                format(transform))
        self._transform = transform

        self._files = [jbfs.jn(o, self.path_infix, s, f)
                       for o in obj_seqs.keys() for s in obj_seqs[o]
                       for f in jbfs.ls(jbfs.jn(path_root, o,
                                                self.path_infix, s), exts=exts)]

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        sample = {'img': jbim.load(jbfs.jn(self._path_root, self._files[idx]))}
        if self._transform:
            sample = self._transform(sample)
        return sample


class PcloudFromDmapAndNormalsSyncedDataset(Dataset):
    """ Loads the depth maps and normal maps. Selects `num_samples` samples from
     both but only within the mask corresponding to nonzero values (nzv) in both
     depth maps and normal maps. The samples are taken randomly, but from the
     same corresponding positions in depth maps and normal maps. If
     `num_samples` > # nzv, # ndz points are returned. If `duplicate_samples`
     is True, then `num_samples` samples are always returned, where the samples
     are duplicated (gradually as they are ordered) if necessary. The samples
     from depth map are reprojected to obtaing a 3D pcloud using intrinsic
     matrix `K`. Hardcoded paths are optimized for the dataset Textureless
     Deformable Surfaces [1] dataset.

    [1] https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/

    Args:
        path_root (str): Root path to the ds.
        obj_seqs (dict): Names of dirs containing object data and seqs.
        K (np.array): Intrinsic camera matrix, shape (3, 3).
        num_samples (int): # samples to include in the point cloud.
        exts_dmap (list of str): Allowed extensions of files with depth maps.
        exts_nmap (list of str): Allowed extensions of files with normal maps.
        transform (callable): Transformation to apply after loading the data.
            If None, no transformation is applied.
        duplicate_samples (bool): Whether to duplicate samples from dmap and
            nmap in case there are not ebough non-zero samples available.
        compute_area (bool): Whether to compute surface area.
    """
    path_dmap_infix = 'depth_maps'
    path_nmap_infix = 'normals'

    def __init__(self, path_root, obj_seqs, K, num_samples, exts_dmap=('npz',),
                 exts_nmap=('npz',), transform='default',
                 duplicate_samples=True, compute_area=False,
                 smooth_normals=False, smooth_normals_k=72, return_dmap=False,
                 return_mesh=False):
        super(PcloudFromDmapAndNormalsSyncedDataset, self).__init__()

        self._path_root = path_root
        self._K = K
        self._num_samples = num_samples
        self._duplicate_samples = duplicate_samples
        self._compute_area = compute_area
        self._smooth_normals = smooth_normals
        self._smooth_normals_k = smooth_normals_k
        self._return_mesh = return_mesh
        self._return_dmap = return_dmap

        self._files_dmaps = \
            [jbfs.jn(o, self.path_dmap_infix, s, f)
             for o in obj_seqs.keys() for s in obj_seqs[o]
             for f in jbfs.ls(jbfs.jn(
                path_root, o, self.path_dmap_infix, s), exts=exts_dmap)]
        self._files_nmaps = \
            [jbfs.jn(o, self.path_nmap_infix, s, f)
             for o in obj_seqs.keys() for s in obj_seqs[o]
             for f in jbfs.ls(jbfs.jn(
                path_root, o, self.path_nmap_infix, s), exts=exts_nmap)]

        assert len(self._files_dmaps) == len(self._files_nmaps)

        if isinstance(transform, str):
            if transform == 'default':
                transform = DataToTensor()
            else:
                raise Exception('Unknown transform option "{}"'.
                                format(transform))
        self._transform = transform

    def __len__(self):
        return len(self._files_dmaps)

    def __getitem__(self, idx):
        dmap = np.load(jbfs.jn(
            self._path_root, self._files_dmaps[idx]))['depth']
        nmap = np.load(jbfs.jn(
            self._path_root, self._files_nmaps[idx]))['normals']

        # Get mask intersecting non-zero values in dmap and normal map.
        mask = jbd.get_mask(dmap) & jbim.get_mask(nmap)
        N = np.sum(mask)

        # Duplicate samples, if not enough non-zero samples available.
        rand_inds = np.random.permutation(N)[:self._num_samples]
        if self._duplicate_samples and N < self._num_samples:
            rand_inds = np.concatenate(
                [rand_inds] * (self._num_samples // N) +
                [rand_inds[:self._num_samples % N]], axis=0)

        # Select non-zero depth values and normals.
        yx = np.stack(np.where(mask), axis=1)[rand_inds]
        pc = jbd.dmap2pcloud(dmap, self._K, yx=yx)
        normals = nmap[yx[:, 0], yx[:, 1]]

        sample = {'pc': pc, 'normals': normals}

        # Compute surface area.
        if self._compute_area:
            A = jbm.area(jbd.dmap2pcloud(dmap, self._K, num_points='all'),
                         jbd.create_topology(dmap))
            sample['area'] = np.array(A, dtype=np.float32)

        # Return mesh.
        if self._return_mesh:
            pc_all = jbd.dmap2pcloud(dmap * mask.astype(np.float32),
                                     self._K, num_points='all')
            faces = jbd.create_topology(dmap * mask.astype(np.float32))
            sample['mesh'] = pc_all
            sample['faces'] = faces

        # Return dmap.
        if self._return_dmap:
            sample['dmap'] = dmap

        return self._transform(sample) if self._transform else sample


class ImgAndPcloudFromDmapAndNormalsSyncedDataset(Dataset):
    """ Loads the RGB image depth maps and normal maps. Class `ImgDataset`
    is used to load an image, class `PcloudFromDmapAndNormalsSyncedDataset`
    is used to load synced normals and pcloud.

    Args:
        path_root (str): Root path to dir containing directories of depth
        obj_seqs (dict): Names of dirs containing object data and seqs.
        K (np.array): Intrinsic camera matrix, shape (3, 3).
        num_samples (int): # samples to include in the point cloud.
        exts_img (list of str): Allowed extensions of image files.
        exts_dmap (list of str): Allowed extensions of dmap files.
        exts_nmap (list of str): Allowed extensions of nmap files.
        tf_img (callable): Tf applied to images. If None, not tf done.
        tf_dmap_nmap (callable): Tf applied to dmaps/nmaps. If None, no tf done.
        duplicate_samples (bool): Whether to duplicate samples from dmap and
            nmap in case there are not ebough non-zero samples available.
        compute_area (bool): Whether to compute surface area.
    """
    def __init__(self, path_root, obj_seqs, K, num_samples, exts_img=('tiff',),
                 exts_dmap=('npz',), exts_nmap=('npz',), tf_img='default',
                 tf_dmap_nmap='default', duplicate_samples=True,
                 compute_area=False, smooth_normals=False,
                 smooth_normals_k=72, return_mesh=False, return_dmap=False):
        super(ImgAndPcloudFromDmapAndNormalsSyncedDataset, self).__init__()

        self._ds_img = ImgDataset(
            path_root, obj_seqs, exts=exts_img, transform=tf_img)
        self._ds_dm_nm = PcloudFromDmapAndNormalsSyncedDataset(
            path_root, obj_seqs, K, num_samples,
            exts_dmap=exts_dmap, exts_nmap=exts_nmap, transform=tf_dmap_nmap,
            duplicate_samples=duplicate_samples, compute_area=compute_area,
            smooth_normals=smooth_normals, smooth_normals_k=smooth_normals_k,
            return_mesh=return_mesh, return_dmap=return_dmap)

        assert len(self._ds_img) == len(self._ds_dm_nm)

    def __len__(self):
        return len(self._ds_img)

    def __getitem__(self, idx):
        """ Returns a dict containing:
            'img' (torch.tensor of float32): Image, shape (C, H, W).
            'pc' (torch.tensor of float32): Pcloud, shape (N, 3).
            'normals' (torch.tensor of float32): Normals, shape (N, 3).
        """
        return {**self._ds_img[idx], **self._ds_dm_nm[idx]}

################################################################################
### Tests
if __name__ == '__main__':
    import jblib.unit_test as jbut

    ############################################################################
    jbut.next_test('ImgAndPcloudFromDmapAndNormalsSyncedDataset')
    import torch
    from torch.utils.data import DataLoader
    from dataset.data_loader import DataLoaderDevice

    path_root = '/cvlabsrc1/cvlab/datasets_jan/texless_defsurf'
    obj_seqs = {
        'cloth': ['Lc_left_edge', 'Lc_tl_te_corns'],
        'tshirt': ['Lc_front'],
        'paper': ['Lc', 'Lc', 'Lr']}

    K = np.loadtxt('/cvlabsrc1/cvlab/datasets_jan/texless_defsurf/camera_intrinsics.txt')
    num_samples = 10000
    dupl_smpls = True
    comp_area = True
    bs = 4
    shuffle = True

    ds = ImgAndPcloudFromDmapAndNormalsSyncedDataset(
        path_root, obj_seqs, K, num_samples, duplicate_samples=dupl_smpls,
        compute_area=comp_area)
    smpl = ds[123]

    pc = smpl['pc']
    normals = smpl['normals']

    assert smpl['img'].shape == (3, 224, 224)
    assert smpl['pc'].shape == (num_samples, 3)
    assert smpl['normals'].shape == (num_samples, 3)

    dl = DataLoaderDevice(
        DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=5,
                   drop_last=True), gpu=True)

    for bi, batch in enumerate(iter(dl)):
        assert batch['img'].shape == (bs, 3, 224, 224) and \
               batch['img'].dtype == torch.float32
        assert batch['pc'].shape == (bs, num_samples, 3) and \
               batch['pc'].dtype == torch.float32
        assert batch['normals'].shape == (bs, num_samples, 3) and \
               batch['normals'].dtype == torch.float32
        assert batch['area'].dtype == torch.float32 and \
               batch['area'].shape == (bs, )

        if bi == 100:
            break
