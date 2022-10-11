import random
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms

class AugmentModule():
    def __init__(self, args, im_size=(32,32)):
        self.set_seed()
        self.model = args.arch
        self.dataset = args.dataset
        self.set_strategy(args.strategy)
        self.set_augmentation(args, im_size)
        self.register_augmentations()

    def register_augmentations(self):
        self.crop = transforms.RandomCrop(size=self.random_crop_size, padding=4)
        self.rotate = transforms.RandomRotation(self.rotate_config)
    
    def set_augmentation(self, args, im_size):
        self.random_crop_size = im_size
        self.noise_config = args.noise
        self.rotate_config = args.rotate
        self.scale_config = args.scale
    
    def set_seed(self, seed=datetime.now()):
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    def set_strategy(self, strategy):
        self.strategy = strategy
        if self.dataset == 'MNIST':
            self.strategy = 'crop_scale_rotate'
        if self.model in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
            self.strategy = 'crop_noise'
    
    def augment(self, images):
        if self.strategy != 'none':
            mean_val = images.view(images.size(0), images.size(1), -1).mean(2).sum(0)
            self.crop.fill = mean_val
            self.rotate.fill = mean_val
            augs = self.strategy.split('-')
            for idx in range(images.size(0)):
                aug = random.choice(augs)
                if aug == 'crop':
                    images[idx] = self.crop(images[idx])
                elif aug == 'noise':
                    images[idx] = self.noise(images[idx])
                elif aug == 'rotate':
                    images[idx] = self.rotate(images[idx])
                elif aug == 'scale':
                    images[idx] = self.scale(images[idx])
        return images
    
    # def diff_augment(x, strategy='', seed = -1, param = None):
    #     if seed == -1:
    #         param.Siamese = False
    #     else:
    #         param.Siamese = True

    #     param.latestseed = seed

    #     if strategy == 'None' or strategy == 'none':
    #         return x

    #     if strategy:
    #         if param.aug_mode == 'M': # original
    #             for p in strategy.split('_'):
    #                 for f in AUGMENT_FNS[p]:
    #                     x = f(x, param)
    #         elif param.aug_mode == 'S':
    #             pbties = strategy.split('_')
    #             set_seed_DiffAug(param)
    #             p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
    #             for f in AUGMENT_FNS[p]:
    #                 x = f(x, param)
    #         else:
    #             exit('unknown augmentation mode: %s'%param.aug_mode)
    #         x = x.contiguous()
    #     return x

    def scale(self, img):
        img_c, img_h, img_w = img[1:]
        h = int((np.random.uniform(1 - self.scale_config, 1 + self.scale_config)) * img_h)
        w = int((np.random.uniform(1 - self.scale_config, 1 + self.scale_config)) * img_w)
        tmp = F.interpolate(img, [h, w], )[0]
        mhw = max(h, w, img_h, img_w)
        im_ = torch.zeros(img_c, mhw, mhw, dtype=torch.float)
        r = int((mhw - h) / 2)
        c = int((mhw - w) / 2)
        im_[:, r:r + h, c:c + w] = tmp
        r = int((mhw - img_h) / 2)
        c = int((mhw - img_w) / 2)
        return im_[:, r:r + img_h, c:c + img_w]

    def noise(self, img):
        return img + self.noise_config * torch.randn(img.shape[1:], dtype=torch.float)
    
    # def diff_rand_scale(self, x, param):
    #     # x>1, max scale
    #     # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    #     ratio = param.ratio_scale
    #     set_seed_DiffAug(param)
    #     sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    #     set_seed_DiffAug(param)
    #     sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    #     theta = [[[sx[i], 0,  0],
    #             [0,  sy[i], 0],] for i in range(x.shape[0])]
    #     theta = torch.tensor(theta, dtype=torch.float)
    #     if param.Siamese: # Siamese augmentation:
    #         theta[:] = theta[0]
    #     grid = F.affine_grid(theta, x.shape).to(x.device)
    #     x = F.grid_sample(x, grid)
    #     return x


    # def diff_rand_rotate(self, x, param): # [-180, 180], 90: anticlockwise 90 degree
    #     ratio = param.ratio_rotate
    #     set_seed_DiffAug(param)
    #     theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    #     theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
    #         [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    #     theta = torch.tensor(theta, dtype=torch.float)
    #     if param.Siamese: # Siamese augmentation:
    #         theta[:] = theta[0]
    #     grid = F.affine_grid(theta, x.shape).to(x.device)
    #     x = F.grid_sample(x, grid)
    #     return x


    # def diff_rand_flip(self, x, param):
    #     prob = param.prob_flip
    #     set_seed_DiffAug(param)
    #     randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    #     if param.Siamese: # Siamese augmentation:
    #         randf[:] = randf[0]
    #     return torch.where(randf < prob, x.flip(3), x)


    # def diff_rand_brightness(self, x, param):
    #     ratio = param.brightness
    #     set_seed_DiffAug(param)
    #     randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    #     if param.Siamese:  # Siamese augmentation:
    #         randb[:] = randb[0]
    #     x = x + (randb - 0.5)*ratio
    #     return x


    # def diff_rand_saturation(self, x, param):
    #     ratio = param.saturation
    #     x_mean = x.mean(dim=1, keepdim=True)
    #     set_seed_DiffAug(param)
    #     rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    #     if param.Siamese:  # Siamese augmentation:
    #         rands[:] = rands[0]
    #     x = (x - x_mean) * (rands * ratio) + x_mean
    #     return x


    # def diff_rand_contrast(self, x, param):
    #     ratio = param.contrast
    #     x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    #     set_seed_DiffAug(param)
    #     randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    #     if param.Siamese:  # Siamese augmentation:
    #         randc[:] = randc[0]
    #     x = (x - x_mean) * (randc + ratio) + x_mean
    #     return x


    # def diff_rand_crop(self, x, param):
    #     # The image is padded on its surrounding and then cropped.
    #     ratio = param.ratio_crop_pad
    #     shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    #     set_seed_DiffAug(param)
    #     translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    #     set_seed_DiffAug(param)
    #     translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    #     if param.Siamese:  # Siamese augmentation:
    #         translation_x[:] = translation_x[0]
    #         translation_y[:] = translation_y[0]
    #     grid_batch, grid_x, grid_y = torch.meshgrid(
    #         torch.arange(x.size(0), dtype=torch.long, device=x.device),
    #         torch.arange(x.size(2), dtype=torch.long, device=x.device),
    #         torch.arange(x.size(3), dtype=torch.long, device=x.device),
    #     )
    #     grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    #     grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    #     x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    #     x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    #     return x


    # def diff_rand_cutout(self, x, param):
    #     ratio = param.ratio_cutout
    #     cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    #     set_seed_DiffAug(param)
    #     offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    #     set_seed_DiffAug(param)
    #     offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    #     if param.Siamese:  # Siamese augmentation:
    #         offset_x[:] = offset_x[0]
    #         offset_y[:] = offset_y[0]
    #     grid_batch, grid_x, grid_y = torch.meshgrid(
    #         torch.arange(x.size(0), dtype=torch.long, device=x.device),
    #         torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
    #         torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    #     )
    #     grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    #     grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    #     mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    #     mask[grid_batch, grid_x, grid_y] = 0
    #     x = x * mask.unsqueeze(1)
    #     return x
    


# class ParamDiffAug():
#     def __init__(self):
#         self.aug_mode = 'S' #'multiple or single'
#         self.prob_flip = 0.5
#         self.ratio_scale = 1.2
#         self.ratio_rotate = 15.0
#         self.ratio_crop_pad = 0.125
#         self.ratio_cutout = 0.5 # the size would be 0.5x0.5
#         self.brightness = 1.0
#         self.saturation = 2.0
#         self.contrast = 0.5




# AUGMENT_FNS = {
#     'color': [rand_brightness, rand_saturation, rand_contrast],
#     'crop': [rand_crop],
#     'cutout': [rand_cutout],
#     'flip': [rand_flip],
#     'scale': [rand_scale],
#     'rotate': [rand_rotate],