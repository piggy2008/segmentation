import os

import torch
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
from datetime import datetime
import custom_model
import torch.nn.functional as F
from ColorToLabel import label2projection_array

# Number of classes in the dataset
num_classes = 14
trial = 'trial_240607_161641'
backbone = 'pidnet_s'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = '../river/val/images'
image_size = 960
bbox_area_th = 50

if 'resnet' in backbone:
    model = custom_model.initialize_model(num_classes, keep_feature_extract=False, backbone=backbone, use_pretrained=False)
else:
    model = custom_model.initialize_model_pidnet(num_classes, backbone=backbone, use_pretrained=False)

state_dict = torch.load(os.path.join('../training_output', trial, 'best_Skydiver.pth'), map_location=device)

model = model.to(device)
model.load_state_dict(state_dict)
model.eval()

transforms_image =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


save_root = os.path.join('../visual_results', trial)
if os.path.exists(save_root):
    current = datetime.now().strftime('%H%M%S')
    save_root = os.path.join('../visual_results', trial + '_' + current)
    os.makedirs(save_root)
else:
    os.makedirs(save_root)
images = os.listdir(root)

for image_path in images:

    image = Image.open(os.path.join(root, image_path))
    show_image = np.asarray(image)
    image_np = np.asarray(image)
    h, w, _ = image_np.shape
    ratio_w = w / image_size
    ratio_h = h / image_size
    # image_np = cv2.resize(image_np, 0.5, 0.5, cv2.INTER_CUBIC)
    # width = int(image_np.shape[1] * 0.3)
    # height = int(image_np.shape[0] * 0.3)
    # dim = (width, height)
    image_np = cv2.resize(image_np, (image_size, image_size), interpolation=cv2.INTER_AREA)

    image = Image.fromarray(image_np)
    image = transforms_image(image)
    image = image.unsqueeze(0)

    image = image.to(device)

    if 'resnet' in backbone:
        outputs = model(image)['out']
    else:
        _, outputs, _ = model(image)
        outputs = F.interpolate(outputs, size=(image_size, image_size), mode='bilinear', align_corners=True)

    _, preds = torch.max(outputs, 1)

    preds = preds.to("cpu")

    preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)
    branch_mask = np.zeros_like(preds_np)
    one_mask = (preds_np == 13)
    branch_mask[one_mask] = 255

    _, _, stats, _ = cv2.connectedComponentsWithStats(branch_mask)

    for stat in stats:
        if stat[4] > bbox_area_th and stat[4] < 900000:
            print('area:', stat[4])
            x = int(stat[0] * ratio_h)
            y = int(stat[1] * ratio_w)
            bbox_h = int((stat[0] + stat[2]) * ratio_h)
            bbox_w = int((stat[1] + stat[3]) * ratio_w)
            cv2.rectangle(show_image, (x, y) , (bbox_h, bbox_w), (255, 0, 0), 3)

    preds_np_color = label2projection_array(preds_np)
    preds_np_color = preds_np_color[:, :, ::-1]
    # preds_np = cv2.cvtColor(preds_np, cv2.COLOR_GRAY2BGR)
    #
    # preds_np_color = cv2.applyColorMap(preds_np * 50, cv2.COLORMAP_HSV)
    #
    cv2.imwrite(os.path.join(save_root, image_path), show_image[:, :, ::-1])
    cv2.imwrite(os.path.join(save_root, image_path[:-4] + '.png'), branch_mask)
    # cv2.imwrite(f"./results/{idx:04}_image.png", image_np)


