# Create gif from individual steps
import argparse

from numpy.lib.function_base import append
import cv2
import glob
import imageio
import numpy as np
import os
import time
from PIL import Image
from tqdm import tqdm


def merge_images(img1, img2, weight):
    '''
    img1: RGB
    img2: RGB
    weight: weight of img1 in overlap 
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


def inpaint(img, mask):
    '''
    From: https://stackoverflow.com/questions/60501986/why-does-the-inpaint-method-not-remove-the-text-from-ic-image
    '''
    res = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir')
    parser.add_argument('--dst_dir')
    parser.add_argument('--blur_kernel_size', type=int)
    parser.add_argument('--fps', type=int)
    args = parser.parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    blur_kernel_size = args.blur_kernel_size
    fps = args.fps

    st_time = time.time()
    path_list = []
    imgs = []
    result = np.zeros_like(cv2.imread(os.path.join(src_dir, f'edge_rgb_blur_{blur_kernel_size}')))
    for k in [blur_kernel_size]:
        dir = os.path.join(src_dir, f'edge_rgb_blur_{k}')
        if not os.path.exists(dir):
            continue
        for name in sorted(os.listdir(dir)):
            path = os.path.join(dir, name)
            if not os.path.exists(path):
                continue
            path_list.append(path)
            result = merge_images(result, cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), 0.0)
            imgs.append(result.copy())

    region_list = [1, 
                    2, 3, 4, 5, 6, 
                    10, 7, 8, 
                    12, 13, 11, 
                    18, 17, 14, 16, 
                    9, 15, 
                    0]
    mask = cv2.imread(f'{dst_dir}/face_parsing_rgb.png')
    mask = cv2.Canny(image=mask, threshold1=100, threshold2=200)     # Canny Edge Detection
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    for i in region_list:
        dir = os.path.join(src_dir, f'face_parsing_{i}')
        if not os.path.exists(dir):
            continue
        for name in sorted(os.listdir(dir)):
            path = os.path.join(dir, name)
            if not os.path.exists(path):
                continue
            path_list.append(path)
            result = merge_images(result, cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), 0.0)
            imgs.append(result.copy())
        result = inpaint(result, mask)
        imgs.append(result.copy())

    # imgs = []
    # result = np.zeros_like(cv2.imread(os.path.join(src_dir, f'edge_rgb_blur_{k}')))
    # for path in tqdm(path_list):
    #     result = merge_images(result, cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), 0.0)
    #     imgs.append(result.copy())
    
    cv2.imwrite(f'{dst_dir}/painting.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    H, W, C = result.shape
    videoWriter = cv2.VideoWriter(f'{dst_dir}/painting.avi', cv2.VideoWriter_fourcc('I','4','2','0'), int(fps), (W, H))
    for img in imgs:
        videoWriter.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    videoWriter.release()
    ed_time = time.time()
    print(f'Save video successfully! {ed_time - st_time}s')
    
    st_time = time.time()
    # gif = imageio.mimsave(f'painting.gif', imgs, 'GIF', fps=fps)      # five times slower than Pillow Image save
    append_images = [Image.fromarray(img) for i, img in enumerate(imgs) if i % int(fps / 10) == 0]
    Image.fromarray(result).save(f'{dst_dir}/painting.gif', save_all=True, loop=0, append_images=append_images, fps=10)
    ed_time = time.time()
    print(f'Save gif successfully! {ed_time - st_time}s')