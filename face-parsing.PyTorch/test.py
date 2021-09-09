#!/usr/bin/python
# -*- encoding: utf-8 -*-

import argparse
import cv2
import numpy as np
import os
import os.path as osp
import torch
import torchvision.transforms as transforms
from logger import setup_logger
from model import BiSeNet
from PIL import Image


def mask_img(img, mask):
    masked_img = img * (mask > 0)
    return masked_img


def vis_parsing_maps(im, parsing_anno, stride, save_im, dst_dir):
    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0],    #background,0
        [255, 85, 0],   #skin,1
        [255, 170, 0],  #r_brow,2
        [255, 0, 85],   #l_brow,3
        [255, 0, 170],  #r_eye,4
        [0, 255, 0],    #l_eye,5
        [85, 255, 0],   #eye_g,6
        [170, 255, 0],  #r_ear,7
        [0, 255, 85],   #l_ear,8
        [0, 255, 170],  #ear_r,9
        [0, 0, 255],    #nose,10
        [85, 0, 255],   #mouth,11
        [170, 0, 255],  #u_lip,12
        [0, 85, 255],   #l_lip,13
        [0, 170, 255],  #neck,14
        [255, 255, 0],  #neck_l,15
        [255, 255, 85], #cloth,16
        [255, 255, 170],#hair,17
        [255, 0, 255],  #hat,18
        [255, 85, 255], #19
        [255, 170, 255],#20
        [0, 255, 255],  #21
        [85, 255, 255], #22
        [170, 255, 255]]#23

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    #num_of_class = np.max(vis_parsing_anno)

    for pi in range(19):
        index = np.where(vis_parsing_anno == pi)
        if index[0].size == 0:
            continue
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
        mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
        mask[index[0], index[1], :] = [1, 1, 1]
        parsing = mask_img(im, mask)
        cv2.imwrite(f'{dst_dir}/face_parsing_{pi}.png', cv2.cvtColor(parsing, cv2.COLOR_RGB2BGR))

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(f'{dst_dir}/face_parsing_gray.png', vis_parsing_anno)
        cv2.imwrite(f'{dst_dir}/face_parsing_rgb.png', vis_parsing_anno_color)
        cv2.imwrite(f'{dst_dir}/face_parsing.png', vis_im)
    # return vis_im


def evaluate(src_path, dst_dir, pth_path='79999_iter.pth'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(pth_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = Image.open(src_path)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        vis_parsing_maps(image, parsing, stride=1, save_im=True, dst_dir=dst_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--result_dir')
    args = parser.parse_args()
    img_path = args.img_path
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    evaluate(src_path=img_path, dst_dir=result_dir)
    # img = cv2.imread(img_path)
    # img = cv2.resize(img, (512, 512))
    # cv2.imwrite(result_dir + '/input.jpg', img)
