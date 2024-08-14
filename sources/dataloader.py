# from __future__ import print_function
# from __future__ import division
import torch
import numpy as np
from torchvision import transforms
import os
import glob
from PIL import Image

class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode, crop_size):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.*'))
        self.img_files.sort()
        self.label_files = []
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            label_filename_with_ext = f"{image_filename}.png"
            self.label_files.append(os.path.join(folder_path, 'labels', label_filename_with_ext))

        # Data augmentation and normalization for training
        # Just normalization for validation
        if "val" == mode :
            self.transforms = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
            ])
        else:
            self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomResizedCrop((512, 512)),
                    transforms.RandomCrop((crop_size, crop_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
                ])

    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]

            image = Image.open(img_path).resize((1024, 1024))
            label = Image.open(label_path).convert('L').resize((1024, 1024), Image.NEAREST)

            # Concatenate image and label, to apply same transformation on both
            image_np = np.asarray(image)
            label_np = np.asarray(label)
            # print(label_path, '----before:', np.unique(label_np))
            new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
            image_and_label_np = np.zeros(new_shape, image_np.dtype)
            image_and_label_np[:, :, 0:3] = image_np
            image_and_label_np[:, :, 3] = label_np

            # Convert to PIL
            image_and_label = Image.fromarray(image_and_label_np)

            # Apply Transforms
            image_and_label = self.transforms(image_and_label)

            # Extract image and label
            image = image_and_label[0:3, :, :]
            label = image_and_label[3, :, :].unsqueeze(0)

            # Normalize back from [0, 1] to [0, 255]
            label = label * 255
            # print('after:', np.unique(label))
            #  Convert to int64 and remove second dimension
            label = label.long().squeeze()

            return image, label

    def __len__(self):
        return len(self.img_files)