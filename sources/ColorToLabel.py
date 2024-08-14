import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

PALETTE = [
    [0, 0, 0], [0, 192, 66], [0, 66, 96], [126, 192, 192],
    [0, 66, 66], [0, 192, 222], [0, 192, 192], [126, 192, 0],
    [0, 192, 96], [18, 192, 66], [126, 18, 192], [0, 0, 204],
    [0, 0, 66], [255, 255, 255]
]

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

def check_label_index(root):
    labels = os.listdir(root)
    indice = []
    for label_path in labels:
        path = os.path.join(root, label_path)
        label_image = Image.open(path).convert('L')

        label_image = np.array(label_image)
        index = np.unique(label_image)
        print(index)
        for i in index:
            if i not in indice:
                indice.append(i)

    print('final:', indice)

def label2projection(root, output):
    if not os.path.exists(output):
        os.makedirs(output)
    labels = os.listdir(root)
    labels.sort()
    for label_path in labels:
        if label_path.endswith('.DS_Store'):
            continue
        path = os.path.join(root, label_path)
        label_image = Image.open(path).convert('L')
        label_image = np.array(label_image)
        mask = np.zeros([label_image.shape[0], label_image.shape[1], 3])
        label_indice = np.unique(label_image)
        print(label_indice)
        for index in label_indice:
            color = PALETTE[index]
            # one_mask = np.where(label_image == index, 1, 0)
            # one_mask = np.stack([one_mask * color[0], one_mask * color[1], one_mask * color[2]], axis=2)
            # mask = np.where(one_mask > 0, one_mask, mask)
            one_mask = (label_image == index)
            mask[one_mask] = color
        mask = Image.fromarray(np.uint8(mask))
        mask_image_path = os.path.join(output, label_path[:-4] + '.jpg')
        mask.save(mask_image_path)

def projection2label(root, output):
    if not os.path.exists(output):
        os.makedirs(output)
    projs = os.listdir(root)
    projs.sort()
    for proj_path in projs:
        if proj_path.endswith('.DS_Store'):
            continue
        path = os.path.join(root, proj_path)
        proj_image = Image.open(path).convert('RGB')
        print(path)
        proj_image = np.array(proj_image)
        label = np.zeros([proj_image.shape[0], proj_image.shape[1]])
        for i, color in enumerate(PALETTE):
            diff = np.linalg.norm(proj_image - np.array(color), axis=2)
            one = diff < 40
            # print(one2.shape)
            label[one] = i
            # label = Image.fromarray(np.uint8(label))
            # label_image_path = os.path.join(output, proj_path[:-4] + '_' + str(i) + '.png')
            # label.save(label_image_path)
            # print(str(i), ':', np.unique(label))
        label = Image.fromarray(np.uint8(label))
        label_image_path = os.path.join(output, proj_path[:-4] + '.png')
        label.save(label_image_path)
        # cv2.imwrite(label_image_path, np.uint8(label))

def label2projection_array(input):

    mask = np.zeros([input.shape[0], input.shape[1], 3])
    label_indice = np.unique(input)
    # print(label_indice)
    for index in label_indice:
        color = PALETTE[index]
        # one_mask = np.where(label_image == index, 1, 0)
        # one_mask = np.stack([one_mask * color[0], one_mask * color[1], one_mask * color[2]], axis=2)
        # mask = np.where(one_mask > 0, one_mask, mask)
        one_mask = (input == index)
        mask[one_mask] = color
    return mask

def projection2label_gray(root, output):
    if not os.path.exists(output):
        os.makedirs(output)
    projs = os.listdir(root)
    projs.sort()
    for proj_path in projs:
        if proj_path.endswith('.DS_Store'):
            continue
        path = os.path.join(root, proj_path)
        proj_image = Image.open(path).convert('L')
        proj_image = np.array(proj_image)
        label = np.zeros([proj_image.shape[0], proj_image.shape[1]])
        for i, color in enumerate(PALETTE):
            mean_color = sum(color) / len(color)
            one = (proj_image == mean_color)
            # print(one2.shape)
            label[one] = i
        print(np.unique(label))
        label = Image.fromarray(np.uint8(label))
        label_image_path = os.path.join(output, proj_path[:-4] + '.png')
        label.save(label_image_path)

def find_river_branch_to_label(root, root_branch, label_root, output):
    if not os.path.exists(output):
        os.makedirs(output)
    projs = os.listdir(root)
    projs.sort()
    for proj_path in projs:
        if proj_path.endswith('.DS_Store'):
            continue
        path = os.path.join(root, proj_path)
        print(path)
        proj_image = Image.open(path).convert('L')
        proj_image = np.array(proj_image)

        path_branch = os.path.join(root_branch, proj_path)
        proj_image_branch = Image.open(path_branch).convert('L')
        proj_image_branch = np.array(proj_image_branch)

        path_label = os.path.join(label_root, proj_path[:-4] + '.png')
        label_image = Image.open(path_label).convert('L')
        label_image = np.array(label_image)

        tmp = abs(proj_image_branch - proj_image)
        # print(tmp.shape)
        # fig, ax = plt.subplots()
        # plt.imshow(tmp, interpolation='none')
        # # ax.format_coord = Formatter(im)
        # plt.colorbar()
        # plt.show()
        one = np.where((tmp > 15) & (tmp <= 250))
        label_image[one] = 13
        # print(np.unique(proj_image))
        output_img = Image.fromarray(label_image)
        output_image_path = os.path.join(output, proj_path[:-4] + '.png')
        output_img.save(output_image_path)

def find_river_branch(root, root_branch, output):
    if not os.path.exists(output):
        os.makedirs(output)
    projs = os.listdir(root)
    projs.sort()
    for proj_path in projs:
        if proj_path.endswith('.DS_Store'):
            continue
        path = os.path.join(root, proj_path)
        print(path)
        proj_image = Image.open(path).convert('L')
        proj_image = np.array(proj_image)

        proj_image_color = Image.open(path).convert('RGB')
        proj_image_color = np.array(proj_image_color)
        # proj_image = cv2.imread(path, 0)
        # print(np.unique(proj_image))
        # print('-----------')
        path_branch = os.path.join(root_branch, proj_path)
        proj_image_branch = Image.open(path_branch).convert('L')
        proj_image_branch = np.array(proj_image_branch)
        # proj_image_branch = cv2.imread(path_branch, 0)

        tmp = abs(proj_image_branch - proj_image)
        print(tmp.shape)
        # fig, ax = plt.subplots()
        # plt.imshow(tmp, interpolation='none')
        # # ax.format_coord = Formatter(im)
        # plt.colorbar()
        # plt.show()
        one = np.where((tmp > 30) & (tmp <= 150))
        proj_image_color[one] = [255, 255, 255]
        # print(np.unique(proj_image))
        output_img = Image.fromarray(proj_image_color)
        output_image_path = os.path.join(output, proj_path)
        output_img.save(output_image_path)

if __name__ == '__main__':
    # root2 = 'val/masks2'
    # output2 = 'val/labels2'
    # projection2label(root2, output2)

    # root = 'val/labels_branch_13'
    # output = 'val/masks3_13'
    # label2projection(root, output)

    root = 'val/masks'
    root_branch = 'val/masks2'
    label_root = 'val/labels'
    output = 'val/labels_branch_13'
    find_river_branch_to_label(root, root_branch, label_root, output)


    # img = Image.open('val/labels3/100_0001_0009.png')
    # plt.imshow(img)
    # plt.show()
    # img2 = Image.open('val/labels2/100_0001_0009.png').convert('L')
    # print(img == img2)
# label = np.zeros([color_image.shape[0], color_image.shape[1]])
# print(len(PALETTE))
#
# for i, color in enumerate(PALETTE):
#     one = (color_image == color).all(axis=2)
#     # print(one2.shape)
#     label[one] = i
#
# print(label.shape)
# print(np.unique(label))