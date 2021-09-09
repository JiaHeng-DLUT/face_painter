import glob
import numpy as np
from PIL import Image


root = 'output/input'
img_path_list = glob.glob(f'{root}/*.png')
dst_path = f'{root}/painting_once.gif'
imgs = [Image.open(path) for path in sorted(img_path_list)]
img = Image.fromarray(np.zeros_like(imgs[0]))
img.save(dst_path, append_images=imgs, save_all=True, loop=0, fps=10)
