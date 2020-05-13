
import h5py
# import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
import os
import cv2
from PIL import Image
import random
# import ipdb

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        # print(files)
        # ipdb.set_trace()

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("input", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("gt", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('gt'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            for dname, ds in h5_file.items():
                # if data is not loaded its cache index is -1
                idx = -1
                if load_data:
                    # add data to the data cache
                    idx = self._add_to_cache(ds.value, file_path)
                
                # type is derived from the name of the dataset; we expect the dataset
                # name to have a name such as 'data' or 'label' to identify its type
                # we also store the shape of the data in case we need it
                self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})
                # print(self.data_info)

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for dname, ds in h5_file.items():
                # add data to the data cache and retrieve
                # the cache index
                idx = self._add_to_cache(ds.value, file_path)

                # find the beginning index of the hdf5 file we are looking for
                file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                # the data info should have the same index since we loaded it in the same way
                self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]

# class MyDataset(torch.utils.Dataset):
#     def __init__(self):
#         self.data_files = os.listdir('data_dir')
#         sort(self.data_files)

#     def __getindex__(self, idx):
#         return load_file(self.data_files[idx])

#     def __len__(self):
#         return len(self.data_files)


# dset = MyDataset()
# loader = torch.utils.DataLoader(dset, num_workers=8)

class TecoGANDataset(data.Dataset):
    def __init__(self, data_root: str, codec:str, qp: int, seq_len:int, crop_size:int, transform=None):
        self.data_root = data_root
        self.lr_dir = os.path.join(self.data_root, "train_compressed_video_frames_"  + codec +  "_" + "qp" + str(qp))
        self.hr_dir = os.path.join(self.data_root, "train_video_frames" )
        self.items = [] # should be  N by sequence
        self.scene_folder_list = []
        for scene_folder in os.listdir(self.lr_dir):
            # print(scene_folder)
            self.scene_folder_list.append(scene_folder)
        
        self.scene_folder_list = sorted(self.scene_folder_list, key = lambda k: int(k.split("_")[-1]))
        # print(self.scene_folder_list)
        for scene_folder in self.scene_folder_list:
            images_list = os.listdir(os.path.join(self.lr_dir, scene_folder))
            image_prefix = "col_high_"
            for n in range(len(images_list) - seq_len):
                item_list = []
                for i in range(seq_len):
                    img_file_name = "col_high_%04d.png" % (i)
                    item_list.append(os.path.join(scene_folder, img_file_name))
                self.items.append(item_list)

        self.qp = qp
        self.codec = codec
        self.crop_size = crop_size
        # self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def image_loader(self, dirpath, file_list, transformation_index = None, rotation=0):
        assert 0 <= rotation <=15
        # print("rotate by " , rotation)
        output_list = []
        h = -1
        w = -1
        c = -1
        # print("transformation_index: ", transformation_index)
        for img_file in file_list:
            img_file_path = os.path.join(dirpath, img_file)
            img_pil = Image.open(img_file_path)
            if  not(transformation_index == -1):
                # print("transpose")
                img_pil = img_pil.transpose(transformation_index)
            if abs(int(rotation)) > 0:
                img_pil = img_pil.rotate(rotation)
            img_np = np.asarray(img_pil)
            # img_np = np.asarray(cv2.imread(img_file_path))
            h,w,c = img_np.shape
            output_list.append(img_np)
        return np.asarray(output_list), (h,w,c)

    def __getitem__(self, item):
        item_list = self.items[item]

        # flip and rotate
        transformation_index_list = [-1, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        # print(transformation_index_list)
        transformation_index = random.choice(transformation_index_list)
        # rotation_angle = np.random.randint(low=0,high=15)
        rotation_angle = 0
        image, img_size = self.image_loader(self.lr_dir, item_list, transformation_index, rotation_angle)
        label, label_size = self.image_loader(self.hr_dir, item_list, transformation_index, rotation_angle)
        # print(image.shape)
        # print(img_size)
        # print(np.transpose(image, (0,3,1,2)).shape)

        # crop the image
        # create a valid index
        h,w,c = img_size
        h_max = h // self.crop_size
        w_max = w // self.crop_size

        valid_h = np.random.randint(low=0, high=h_max)
        valid_w = np.random.randint(low=0, high=w_max)
        start_h = valid_h * self.crop_size
        end_h = start_h + self.crop_size
        start_w = valid_w * self.crop_size
        end_w = start_w + self.crop_size
        image = image[:,start_h:end_h, start_w:end_w, :]
        label = label[:,start_h:end_h, start_w:end_w, :]
        image = np.transpose(image, (0,3,1,2)) / 255.0
        label = np.transpose(label, (0,3,1,2)) / 255.0
        # filename = "image_" + str(item) + '.png'
        # cv2.imwrite(filename, image[0])

        return image, label


if __name__ == "__main__":
    data_root = "/media/data3/tjtanaa/tecogan_video_data"
    codec = "libx264"
    qp = "37"
    seq_len = 5
    crop_size = 512
    dataset = TecoGANDataset(data_root, codec, qp, seq_len, crop_size)
    dataset[0]


# def get_local_dataloaders(local_data_root: str, batch_size: int = 8, transform: Callable = None,
#                           test_ratio: float = 0.1, num_workers: int = 8) -> Dict[str, torch.utils.data.DataLoader]:
#     # Local training
#     local_items = list_items_local(os.path.join(local_data_root, "images"))
#     dataset = ImageDataset(local_data_root, local_items, transform=transform)

#     # Split using consistent hashing
#     train_indices, test_indices = consistent_train_test_split(local_items, test_ratio)
#     return {
#         "train": torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices),
#                                              num_workers=num_workers),
#         "test": torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_indices),
#                                             num_workers=num_workers)
#     }