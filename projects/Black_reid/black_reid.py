import glob
import re
import pdb
import os.path as osp
import os
from .bases import ImageDataset
import warnings
from fastreid.data.datasets import DATASET_REGISTRY


__all__ = ['Black_reid']


@DATASET_REGISTRY.register()
class Black_reid(ImageDataset):

    def __init__(self, cfg):
        self.dataset_dir = cfg.DATASETS.DATASETS_ROOT
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, mode='train', relabel=True)
        query = self.process_dir(self.query_dir, mode='query', relabel=False)
        gallery = self.process_dir(self.gallery_dir, mode='gallery', relabel=False)

        super(Black_reid, self).__init__(train, query, gallery)

    def process_dir(self, dir_path, mode, relabel=False):

        img_paths = glob.glob(osp.join(dir_path, '*g'))
        pattern = re.compile(r'(.*\d.*)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ =  pattern.search(img_path).groups()
            pid = pid.split('/')[-1]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pids, camid = pattern.search(img_path).groups()
            pids = pids.split('/')[-1]
            pid = pids
            if relabel:
                pid = pid2label[pids]
            else:
                if pids.startswith('b'):
                    pid = pids.split('_')[1]
                else:
                    pid = pids
            if pids.startswith('b'):
                black_id = 1
            else:
                black_id = 0
            pid = int(pid)
            camid = int(camid)
            black_id = int(black_id)

            if mode == 'train':
                data.append((img_path, pid, camid, black_id))
            else:
                data.append((img_path, pid, camid))

        return data
