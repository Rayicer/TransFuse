import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.TransFuse import TransFuse_S, TransFuse_L, TransFuse_L_384
from utils.dataloader import test_dataset
import imageio


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='snapshots/TransFuse-19_best.pth')
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default=None, help='path to save inference segmentation')

    opt = parser.parse_args()

    model = TransFuse_S().cuda()
    model.load_state_dict(torch.load(opt.ckpt_path))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    print('evaluating model: ', opt.ckpt_path)

    image_root = '{}/data_test.npy'.format(opt.test_path)
    gt_root = '{}/mask_test.npy'.format(opt.test_path)
    test_loader = test_dataset(image_root, gt_root)

    dice_bank = []
    iou_bank = []
    acc_bank = []

    for i in range(test_loader.size):
        image, gt = test_loader.load_data()
        gt = 1*(gt>0.5)
        image = image.cuda()

        with torch.no_grad():
            _, _, res = model(image)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 1*(res > 0.5)

        if opt.save_path is not None:
            imageio.imwrite(opt.save_path+'/'+str(i)+'_pred.jpg', res)
            imageio.imwrite(opt.save_path+'/'+str(i)+'_gt.jpg', gt)

        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)

    print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
        format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
