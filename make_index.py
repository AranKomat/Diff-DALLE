import argparse
import os
import json
from PIL import Image
import multiprocessing
from diff_dalle.datasets import _list_text_files_recursively

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='')
parser.add_argument('--img_min_sizes', default='0')
parser.add_argument('--index_path_save', default='')
parser.add_argument('--index_path_load', default='')
args = parser.parse_args()
min_sizes = args.img_min_sizes.split(',')

if args.index_path_load:
    with open(args.index_path_load) as f:
        data = json.load(f)
else:
    data = _list_text_files_recursively(args.data_path)
    with open(args.index_path_save, 'w') as f:
        json.dump(data, f, indent=4) 

def fun(data):
    new_data = {k: [] for k in min_sizes}
    img = os.path.splitext(data)[0].replace('/texts/', '/images/')  
    for ext in [".jpg", ".jpeg", ".png", ".gif"]:
        cur_path = img + ext
        if os.path.exists(cur_path):
            try:
                with Image.open(cur_path) as img:
                    width, height = img.size
                size = min(width, height)
                for min_size in min_sizes:
                    print(min_size, size)
                    if size >= int(min_size):
                        new_data[min_size] += [data]
            except:
                pass
            break
    return new_data


with multiprocessing.Pool(processes=os.cpu_count()) as pool:
    results = pool.map(fun, data)

new_results = {k: [] for k in min_sizes}
for d in results:
    for min_size in min_sizes:
        new_results[min_size] += d[min_size]
    
for min_size in min_sizes:
    path = os.path.splitext(args.index_path_save)[0] + '-' + str(min_size) + ".json"
    with open(path, 'w') as f:
        json.dump(new_results[min_size], f, indent=4) 