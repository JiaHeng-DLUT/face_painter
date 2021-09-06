import os
import time
import shutil


def get_time_str():
    '''
    From https://github.com/xinntao/EDVR/blob/b02e63a0fc5854cad7b11a87f74601612e356eff/basicsr/utils/misc.py#L21
    '''
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def rename(path):
    """If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    
    From: https://github.com/xinntao/EDVR/blob/b02e63a0fc5854cad7b11a87f74601612e356eff/basicsr/utils/misc.py#L25
    """
    if os.path.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)


st_time = time.time()
img_path = 'input.jpg'
result_dir = 'output'
blur_kernel_size = 5
fps = 200


if os.path.exists(result_dir):
    rename(result_dir)
print(f'================ 1. Detect edges -> {result_dir} ================')
cmd = f'python detect.py --img_path {img_path} --result_dir {result_dir} --blur_kernel_size {blur_kernel_size}'
# print(cmd)
os.system(cmd)
print(f'================ 2. Parse face -> {result_dir} ================')
cmd = f'cd face-parsing.PyTorch && python test.py --img_path ../{img_path} --result_dir ../{result_dir}'
# print(cmd)
os.system(cmd)
print(f'================ 3. Paint face parsing maps -> {result_dir} ================')
cmd = f'cd PaintTransformer_backup/inference && python inference.py --src_dir ../../{result_dir} --result_dir ../../{result_dir}'
# print(cmd)
os.system(cmd)
print(f'================ 4. Make gif and video -> {result_dir} ================')
cmd = f'python make_gif.py --src_dir {result_dir} --dst_dir {result_dir} --blur_kernel_size {blur_kernel_size} --fps {fps}'
# print(cmd)
os.system(cmd)
ed_time = time.time()
print(f'================ Done! {ed_time - st_time}s ================')