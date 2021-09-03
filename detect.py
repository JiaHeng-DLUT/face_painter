'''
Use Paint Transformer to paint face edges.
'''
import argparse
import cv2
import numpy as np
import os


def main(path, result_dir):
    max_blur_kernel_size = 15
    blur_kernel_size_list = [2 * k + 1 for k in range(max_blur_kernel_size)]
    blur_kernel_size_list.reverse()
    img = cv2.imread(path)
    H, W = img.shape[:2]
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i, k in enumerate(blur_kernel_size_list):
        # ================ 1. Detect face edges. ================ #
        # Not necessary to convert RGB to gray.
        # img_blur = cv2.GaussianBlur(img_gray, (k, k), 0)
        img_blur = cv2.GaussianBlur(img, (k, k), 0)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)   # Canny Edge Detection
        cv2.imwrite(f'{result_dir}/edge_{i}.png', edges)

        # ================ 2. Mask face edges. ================ #
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_rgb = img * (edges > 0)
        cv2.imwrite(f'{result_dir}/edge_rgb_{i}.png', edges_rgb)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--result_dir')
    args = parser.parse_args()
    img_path = args.img_path
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    main(img_path, result_dir)
