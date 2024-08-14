import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import argparse
import pathlib
from datetime import datetime
import json

# Local import
from dataloader import DataLoaderSegmentation
from custom_model import initialize_model, initialize_model_pidnet
from train import train_model

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

"""
    Version requirements:
        PyTorch Version:  1.4.0
        Torchvision Version:  0.5.0
"""


def main(data_dir, dest_dir, num_classes, batch_size, crop_size, num_epochs, save_epoch, keep_feature_extract, backbone, weight, device, model_path):
# def main():

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: DataLoaderSegmentation(os.path.join(data_dir, x), x, crop_size) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train', 'val']}

    print("Initializing Model...")

    # Initialize model
    if 'resnet' in backbone:
        model = initialize_model(num_classes, keep_feature_extract, backbone, use_pretrained=True)
    else:
        model = initialize_model_pidnet(num_classes, backbone, use_pretrained=keep_feature_extract, model_path=model_path)

    # Detect if we have a GPU available
    device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    if keep_feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss(weight=(torch.FloatTensor(weight).to(device) if weight else None))

    # Prepare output directory
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    print("Train...")

    # Train and evaluate
    model_dict, hist = train_model(model, num_classes, dataloaders_dict, criterion, optimizer_ft,
                                   device, dest_dir, num_epochs=num_epochs, save_epoch=save_epoch, model_name=backbone)

    print("Save ...")
    torch.save(model_dict, os.path.join(dest_dir, "best_Skydiver.pth"))


def args_preprocess():
    # Command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data_dir", default='./sample_dataset', help='Specify the dataset directory path, should contain train/Images, train/Labels, val/Images and val/Labels')
    # parser.add_argument(
    #     "--dest_dir", default='./output', help='Specify the  directory where model weights shall be stored.')
    # parser.add_argument("--num_classes", default=5, type=int, help="Number of classes in the dataset, index 0 for no-label should be included in the count")
    # parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train for")
    # parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training (change depending on how much memory you have)")
    # parser.add_argument("--keep_feature_extract", action="store_true", help="Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params")
    # parser.add_argument('-w', action='append', type=float, help="Add more weight to some classes. If this argument is used, then it should be called as many times as there are classes (see --num_classes)")
    #
    current = datetime.now().strftime('%y%m%d_%H%M%S')
    hyper_param = {
        'data_dir': '../river',
        'output_dir': '../training_output/trial_' + current,
        'num_classes': 14,
        'epochs': 2000,
        'save_epoch': 100,
        'batch_size': 24,
        'crop_size': 960,
        'keep_feature_extract': False,
        'backbone': 'pidnet_l',
        'device': 2,
        'model_path': '../pre_trained/PIDNet_L_Cityscapes_test.pt'
    }

    # args = parser.parse_args()
    # args.crop_size = 960
    if not os.path.exists(hyper_param['output_dir']):
        os.makedirs(hyper_param['output_dir'])
    json.dump(hyper_param, open(os.path.join(hyper_param['output_dir'], 'parameters.json'), 'w'))
    # Build weight list
    weight = []
    # if args.w:
    #     for w in args.w:
    #         weight.append(w)

    main(hyper_param['data_dir'], hyper_param['output_dir'],
         hyper_param['num_classes'], hyper_param['batch_size'], hyper_param['crop_size'],
         hyper_param['epochs'], hyper_param['save_epoch'], hyper_param['keep_feature_extract'],
         hyper_param['backbone'], weight, hyper_param['device'], hyper_param['model_path'])


if __name__ == '__main__':
    args_preprocess()