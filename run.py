import os

img_path = '/home/jh/jiaheng-data/face_painter/input_159.jpg'
inter_result_dir = '/home/jh/jiaheng-data/face_painter/inter_output'
result_dir = '/home/jh/jiaheng-data/face_painter/output'
# 1. Detect edges -> inter_output/
cmd = f'python detect.py --img_path {img_path} --result_dir {inter_result_dir}'
print(cmd)
#os.system(cmd)
# 2. Parse face -> inter_output/
cmd = f'cd face-parsing.PyTorch && python test.py --img_path {img_path} --result_dir {inter_result_dir}'
print(cmd)
#os.system(cmd)
# 3. Generate intermediate results -> output/
cmd = f'cd PaintTransformer_backup/inference && python inference.py --src_dir {inter_result_dir} --result_dir {result_dir}'
print(cmd)
os.system(cmd)
# 4. make gif -> output/painting.gif
cmd = f'python make_gif.py --src_dir {result_dir}' 
print(cmd)
os.system(cmd)