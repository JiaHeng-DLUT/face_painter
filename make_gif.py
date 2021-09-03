# Create gif from individual steps
import argparse
import cv2
import glob
import imageio
import numpy as np
import os
from tqdm import tqdm


def merge_images(img1, img2, weight):
    '''
    img1: RGB
    img2: RGB
    w: weight of img1 in overlap 
    '''
    w1 = img1.sum(axis=-1)
    w2 = img2.sum(axis=-1)
    w1 = (w1 != 0)
    w2 = (w2 != 0)
    w = (w1 & w2)
    w1 = w1 - (w * (1 - weight))
    w2 = w2 - (w * weight)
    w1 = np.stack([w1, w1, w1], axis=-1)
    w2 = np.stack([w2, w2, w2], axis=-1)
    img = img1 * w1 + img2 * w2
    return img.astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir')
    # parser.add_argument('--weight')
    args = parser.parse_args()
    src_dir = args.src_dir
    # weight = args.weight
    path_list = []
    # region_list = [17, 5, 4, 3, 2, 10, 12, 13, 1, 8, 7, 16, 6, 9, 11, 14, 15, 18, 19]  #先画五官
    region_list = [1, 17, 5, 4, 3, 2, 10, 12, 13, 8, 7, 16, 6, 9, 11, 14, 15, 18, 19]  #先画脸皮
    imgs = []
    result = np.zeros_like(cv2.imread(os.path.join(src_dir, 'face_parsing.png')))
    H, W, C = result.shape
    imgs.append(result)
    for i in [5]:
        dir = os.path.join(src_dir, f'edge_rgb_{i}')
        if not os.path.exists(dir):
            continue
        for name in os.listdir(dir):
            path = os.path.join(dir, name)
            if not os.path.exists(path):
                continue
            path_list.append(path)
    for i in region_list:
        dir = os.path.join(src_dir, f'face_parsing_{i}')
        if not os.path.exists(dir):
            continue
        for name in os.listdir(dir):
            path = os.path.join(dir, name)
            if not os.path.exists(path):
                continue
            path_list.append(path)
    for path in tqdm(path_list):
        result = merge_images(result, cv2.cvtColor(cv2.cv2.imread(path), cv2.COLOR_BGR2RGB), 0.0)
        imgs.append(result.copy())
    gif = imageio.mimsave(f'painting.gif', imgs, 'GIF', duration=0.01)
    videoWriter = cv2.VideoWriter(f'painting.avi', cv2.VideoWriter_fourcc('I','4','2','0'), 30, (W, H))
    for img in imgs:
        videoWriter.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    videoWriter.release()