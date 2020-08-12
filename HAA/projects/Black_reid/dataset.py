import torch
from torch.utils.data import Dataset
from fastreid.data.data_utils import read_image
import pdb
import numpy as np
class BlackreidDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, mode='train', relabel=True):
        self.transform = transform
        self.relabel = relabel
        self.mode = mode
        self.pid_dict = {}
        if self.relabel:
            self.img_items = []
            pids = set()
            for i, item in enumerate(img_items):
                pid = self.get_pids(item[0], item[1])
                self.img_items.append((item[0], pid, item[2], item[3]))  # replace pid
                pids.add(pid)
            self.pids = pids
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
        else:
            self.img_items = img_items
            
    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path, pid, camid, black_id = self.img_items[index]
            img = read_image(img_path)
            img_array = np.array(img)
            if self.transform is not None:   
                img = self.transform(img)
            if self.relabel:            
                pid = self.pid_dict[pid]
            return {
            'images': img,
            'targets': pid,
            'camid': camid,
            'black_id': black_id
            }
        else:
            img_path, pid, camid = self.img_items[index]
            img = read_image(img_path)
            if self.transform is not None:   
                img = self.transform(img)
            if self.relabel:            
                pid = self.pid2label[pid]
            return {
            'images': img,
            'targets': pid,
            'camid': camid,
            'img_path': img_path
            }


    def get_pids(self, file_path, pid):
        """ Suitable for muilti-dataset training """
        if 'cuhk03' in file_path:
            prefix = 'cuhk'
        else:
            prefix = file_path.split('/')[1]
        return prefix + '_' + str(pid)

    def update_pid_dict(self, pid_dict):
        self.pid_dict = pid_dict


class data_prefetcher():
    def __init__(self, cfg, loader):
        self.loader = loader
        self.loader_iter = iter(loader)

        # normalize
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        self.mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, num_channels, 1, 1)
        self.std = torch.tensor(cfg.MODEL.PIXEL_STD).view(1, num_channels, 1, 1)
        self.preload()

    def reset(self):
        self.loader_iter = iter(self.loader)
        self.preload()

    def preload(self):
        try:
            self.next_inputs = next(self.loader_iter)
        except StopIteration:
            self.next_inputs = None
            return

        self.next_inputs["images"].sub_(self.mean).div_(self.std)

    def next(self):
        inputs = self.next_inputs
        self.preload()
        return inputs
