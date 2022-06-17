import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import glob

class FlickrDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 512
        opt.crop_size = 512
        opt.label_nc = 28
        opt.contain_dontcare_label = True
        opt.semantic_nc = 29 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])
        image, label = self.transforms(image, label)
        label = label * 1.0
        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):
        mode = self.opt.phase
        if self.for_metrics:
            mode = 'valid'
        images = glob.glob(os.path.join(self.opt.dataroot, mode, "imgs") + "/*.jpg")
        images = list(sorted(images))
        labels = glob.glob(os.path.join(self.opt.dataroot, mode, "labels") + "/*.png")
        labels = list(sorted(labels))
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert os.path.basename(images[i]).replace(".jpg", "") == \
                os.path.basename(labels[i]).replace(".png", ""),\
                '%s and %s are not matching' % (images[i], labels[i])
        return images, labels

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label
