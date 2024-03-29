

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x
        self.resize = transforms.Resize((480, 640))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': "ETH3D"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class ETH3D(Dataset):
    def __init__(self, data_dir_root):
        import glob

        # image paths are of the form <data_dir_root>/{HR, LR}/<scene>/{color, depth_filled}/*.png
        self.image_files = glob.glob(os.path.join(
            data_dir_root, "rgb", '*.png'))
        self.depth_files = [r.replace("rgb", "gt") for r in self.image_files]
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path),
                           dtype='uint16').astype('float')  * 100/ 65535  # mm to meters

        # print(depth[160][224])

        # print(np.shape(image))
        # print(np.shape(depth))

        # depth[depth > 8] = -1
        depth = depth[..., None]

        sample = dict(image=image, depth=depth)

        # return sample
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

       

        return sample

    def __len__(self):
        return len(self.image_files)


def get_eth3d_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = ETH3D(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)

# get_diml_indoor_loader(data_dir_root="datasets/diml/indoor/test/HR")
# get_diml_indoor_loader(data_dir_root="datasets/diml/indoor/test/LR")
