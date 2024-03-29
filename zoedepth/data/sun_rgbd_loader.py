# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import cv2


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x

    def __call__(self, sample):
        image, depth, focal_w, focal_h, mask = sample['image'], sample['depth'], sample['focal_w'], sample['focal_h'],sample['mask']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        return {'image': image, 'depth': depth, 'focal_w': focal_w, 'focal_h': focal_h,'mask': mask, 'dataset': "sunrgbd"}

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


class SunRGBD(Dataset):
    def __init__(self, data_dir_root):
        root_folder = data_dir_root
        # with open(os.path.join(root_folder, "imagelist.txt"), 'r') as f:
        with open(os.path.join(root_folder, "sunrgbd_training_images1.txt"), 'r') as f:
            imglist = f.read().split()
            # focal_temp = imglist[0].split('/')

        samples = []
        for basename in imglist:
            img_path = os.path.join(root_folder, basename)
            focal_temp = basename.split('/')
            focal_temp.pop()
            focal_temp.pop()

            # depthbase = img_path.replace("image", "depth_bfx").replace("jpg", "png")
            temppath = ''
            for path in focal_temp:
                temppath = os.path.join(temppath, path)
            # print(temppath)
            # focalbase = os.path.join(focal_temp[0], focal_temp[1],focal_temp[2],focal_temp[3],"intrinsics.txt")

            # depth_path = os.path.join(root_folder, depthbase)
            focal_path = os.path.join(root_folder, temppath, "intrinsics.txt")


            # depth_path = glob.glob(
            #     os.path.join(root_folder, temppath, 'depth_bfs','*'))
            depth_paths = os.listdir(os.path.join(root_folder, temppath, 'depth_bfx'))
            depth_path = os.path.join(root_folder, temppath, 'depth_bfx', depth_paths[0])
            # depth_paths = os.listdir(os.path.join(root_folder, temppath, 'depth'))
            # depth_path = os.path.join(root_folder, temppath, 'depth', depth_paths[0])

            # print(depth_path)


            # valid_mask_path = os.path.join(
            #     root_folder, 'mask_invalid', basename+".png")
            # transp_mask_path = os.path.join(
            #     root_folder, 'mask_transp', basename+".png")

            samples.append(
                (img_path, depth_path, focal_path))

        self.samples = samples
        self.transform = ToTensor()
        # self.normalize = T.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def __getitem__(self, idx):
        img_path, depth_path, focal_path = self.samples[idx]
        image = Image.open(img_path)
        depth = Image.open(depth_path)
        with open(focal_path, 'r') as f:
            imglist = f.read().split()
            focal = float(imglist[0])

        # focal = image.shape[0]
        # print(depth.size)
        # focal_w = focal
        # focal_h = focal
        focal_w = focal / depth.size[0] * 640
        focal_h = focal / depth.size[1] * 480
        # print("0", depth.size[0])
        # print(depth.size[1])
        image = image.resize((640, 480), Image.NEAREST)
        depth = depth.resize((640, 480), Image.NEAREST)

        #########
        ratio = random.randint(15, 20) / 20
        # x_start = random.randint(0, int((1 - ratio) * 480))
        # y_start = random.randint(0, int((1 - ratio) * 640))
        x_start = 240 - int(ratio * 240)
        y_start = 320 - int(ratio * 320)
        focal_w = focal_w / ratio
        focal_h = focal_h / ratio
        depth = depth.crop(
            (y_start, x_start, y_start + int(ratio * 640), x_start + int(ratio * 480)))
        image = image.crop(
            (y_start, x_start, y_start + int(ratio * 640), x_start + int(ratio * 480)))

        depth = depth.resize((640, 480), Image.NEAREST)
        image = image.resize((640, 480), Image.BILINEAR)
        ############




        random_angle = (random.random() - 0.5) * 2 * 1.0
        image = self.rotate_image(image, random_angle)
        depth = self.rotate_image(depth, random_angle, flag=Image.NEAREST)


        # image = np.asarray(Image.open(img_path), dtype=np.float32) / 255.0
        # depth = np.asarray(Image.open(depth_path), dtype='uint16') / 10000.0

        image = np.asarray(image, dtype=np.float32) / 255.0

        depth = np.asarray(depth, dtype='uint16') / 10000.0

        mask = np.logical_and(depth > 1e-2,
                              depth < 8).squeeze()[None, ...]


        #######depaug
        trans_depth = random.randint(1, 10)
        if trans_depth > 7:
            focal_w = focal_w * ratio
            focal_h = focal_h * ratio
            depth = depth * ratio
        ###########

        # mask = np.logical_and(depth > 1e-3,
        #                       depth < 10).squeeze()[None, ...]



        # depth[depth > 8] = -1
        depth = depth[..., None]



        return self.transform(dict(image=image, depth=depth, focal_w = focal_w, focal_h = focal_h, mask = mask))


    #####aug
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        # print("sunrotato")
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth

    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug



    #####





    def __len__(self):
        return len(self.samples)


def get_sunrgbd_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = SunRGBD(data_dir_root)
    return DataLoader(dataset, batch_size, shuffle=True, **kwargs)
