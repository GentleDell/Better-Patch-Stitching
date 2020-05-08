""" Data loader of adjusted ShapeNet dataset. Code adapted from [1].

[1] Groueix Thibault et al. AtlasNet: A Papier-Mâché Approach to Learning 3D
    Surface Generation. CVPR 2018.

Author: Jan Bednarik, jan.bednarik@epfl.ch
Date: 17.2.2020
"""

# Python std.
import os
import random

# 3rd party.
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Project files.
from helpers import Device, jn


CHUNK_SIZE = 150
lenght_line = 60


def my_get_n_random_lines(path, n=5):
    MY_CHUNK_SIZE = lenght_line * (n+2)
    lenght = os.stat(path).st_size
    with open(path, 'r') as file:
        file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
        chunk = file.read(MY_CHUNK_SIZE)
        lines = chunk.split(os.linesep)
    return lines[1:n+1]


class ShapeNet(Dataset):
    """
    Args:
        rootimg:
        rootpc:
        class_choice:
        train:
        npoints:
        gen_view:
        SVR:
        idx:
        load_area (bool): Whether to return area [m^2] of every mesh.
    """

    def __init__(self, rootimg, rootpc, path_category_file,
                 class_choice='chair', train=True, npoints=2500,
                 gen_view=False, SVR=False, idx=0, load_area=False):
        super(ShapeNet, self).__init__()
        self.train = train
        self.rootimg = rootimg
        self.rootpc = rootpc
        self.npoints = npoints
        self.datapath = []
        self.catfile = path_category_file
        self.cat = {}
        self.meta = {}
        self.SVR = SVR
        self.gen_view = gen_view
        self.idx = idx
        self._load_area = load_area
        self._zero = torch.tensor(0., dtype=torch.float32)

        # Find all category names in the dataset.
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # Keep only selected object categories.
        if class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items()
                        if k in class_choice}
        print(self.cat)

        empty = []
        for item in self.cat:
            # Get image file names.
            dir_img = os.path.join(self.rootimg, self.cat[item])
            fns_img = sorted(os.listdir(dir_img))

            # Get pts file names.
            try:
                dir_point = os.path.join(self.rootpc, self.cat[item], 'ply')
                fns_pc = sorted(os.listdir(dir_point))
            except:
                fns_pc = []

            # Select only those file names present both in imgs and pts dirs.
            fns = [val for val in fns_img if val + '.points.ply' in fns_pc]
            print('category ', self.cat[item], 'files ' + str(len(fns)),
                  len(fns) / float(len(fns_img)), "%"),

            # Select train/valid subset.
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]

            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    objpath = jn(
                        '/cvlabdata2/cvlab/datasets_jan/shapenet/'
                        'ShapeNetCore.v2', self.cat[item], fn,
                        'models/model_normalized.obj')
                    self.meta[item].append(
                        (os.path.join(dir_img, fn, 'rendering'),
                         os.path.join(dir_point, fn + '.points.ply'),
                         os.path.join(dir_point, fn + '.points.ply2.txt'),
                         item, objpath, fn))
            else:
                empty.append(item)
        for item in empty:
            del self.cat[item]
        self.idx2cat = {}
        self.size = {}
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            for fn in self.meta[item]:
                self.datapath.append(fn)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor(),
            # normalize,
        ])

        # RandomResizedCrop or RandomCrop
        self.dataAugmentation = transforms.Compose([
            transforms.RandomCrop(127),
            transforms.RandomHorizontalFlip(),
        ])
        self.validating = transforms.Compose([
            transforms.CenterCrop(127),
        ])

    def __getitem__(self, index):
        def load_tf_and_area(pth):
            with open(pth, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3
                T = np.array([float(v) for v in lines[0].split()],
                             dtype=np.float32)
                s = float(lines[1])
                a = float(lines[2])
            return T, s, a

        fn = self.datapath[index]
        with open(fn[1]) as fp:
            for i, line in enumerate(fp):
                if i == 2:
                    try:
                        lenght = int(line.split()[2])
                    except ValueError:
                        print(fn)
                        print(line)
                    break
        # this for loop is because of some weird error that happens sometime
        # during loading I didn't track it down and brute force the solution
        # like this.
        for i in range(15):
            try:
                mystring = my_get_n_random_lines(fn[1], n=self.npoints)
                point_set = np.loadtxt(mystring).astype(np.float32)
                break
            except ValueError as excep:
                print(fn)
                print(excep)

        # Extract pcloud and normals normalized to unit length.
        pc_gt = torch.from_numpy(point_set[:, :3])
        normals = torch.from_numpy(
            point_set[:, 3:] /
            np.linalg.norm(point_set[:, 3:], axis=1, keepdims=True))

        # load image
        if self.SVR:
            if self.train:
                N_tot = len(os.listdir(fn[0])) - 3
                if N_tot == 1:
                    print("only one view in ", fn)
                if self.gen_view:
                    N = 0
                else:
                    N = np.random.randint(1, N_tot)
                if N < 10:
                    im = Image.open(os.path.join(fn[0], "0" + str(N) + ".png"))
                else:
                    im = Image.open(os.path.join(fn[0], str(N) + ".png"))

                im = self.dataAugmentation(im)  # random crop
            else:
                if self.idx < 10:
                    im = Image.open(os.path.join(fn[0], "0" + str(self.idx) +
                                                 ".png"))
                else:
                    im = Image.open(os.path.join(fn[0], str(self.idx) + ".png"))
                im = self.validating(im)  # center crop
            data = self.transforms(im)  # scale
            data = data[:3, :, :]
        else:
            data = self._zero

        sample = {'img': data, 'pcloud': pc_gt, 'normals': normals}

        # Compute area and store.
        if self._load_area:
            sample['area'] = torch.from_numpy(
                np.array(load_tf_and_area(fn[2])[2], dtype=np.float32))

        return sample

    def __len__(self):
        return len(self.datapath)


class DataLoaderDevice(Device):
    """ Wrapper of torch.utils.data.DataLoader which automatically places
    the loaded batches of tensors to a given device.

    Args:
        dl (torch.utils.data.DataLoader): Data loader.
        gpu (bool): Whether to use gpu.
    """
    def __init__(self, dl, gpu=True):
        super(DataLoaderDevice, self).__init__(gpu=gpu)
        self._dl = dl

    def __len__(self):
        return len(self._dl)

    def __iter__(self):
        batches = iter(self._dl)
        for batch in batches:
            if isinstance(batch, dict):
                yield {k: v.to(self.device) for k, v in batch.items()}
            elif not (isinstance(batch, tuple) or isinstance(batch, list)):
                yield batch.to(self.device)
            else:
                yield tuple([d.to(self.device) for d in batch])
