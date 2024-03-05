from streetview import get_panorama
import multiprocessing
import argparse
import os
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser('Paths info and configurations', add_help=False)
parser.add_argument('--ids_file_path', type=str, required=True, help='Path to the file with the list of panorama IDs')
parser.add_argument('--images_save_path', type=str, required=True, help='Dircetory path to store the images')
parser.add_argument('--num_processes', type=int, default=1, help='Number of parallel processes to download images')
parser.add_argument('--percentage', type=float, default=1.0, help='Percentage of images to download')
parser.add_argument('--dont_preprocess', action='store_true', help='Do not remove top and bottom artifacts')

args = parser.parse_args()

SAVE_PATH = args.images_save_path
TYPICAL_WIDTH = 13312
TYPICAL_HEIGHT = 6656
CROP_HEIGHT_INDEX = 1000
POST_HEIGHT = 4000

assert args.num_processes>=0, f'The inputed number of precesses ({args.num_processes}) must be a positive number'
assert args.num_processes<=multiprocessing.cpu_count(), f'The inputed number of precesses ({args.num_processes}) is higher than the number of cpus available ({multiprocessing.cpu_count()})'

os.makedirs(args.images_save_path, exist_ok=True)

def download_panoramas(panorama_file_line: str):
    panorama_id, year, latitude, longitude = panorama_file_line.strip().split(' ')
    image = get_panorama(pano_id=panorama_id)
    image_name = '_'.join([panorama_id,latitude,longitude,year])

    if np.sum(np.array(image))==0:
        print(f'Could not get image {image_name}, a txt file with its name will be stored instead.')
        with open(os.path.join(SAVE_PATH, image_name + '.txt'), 'w') as tx:
            tx.write(panorama_file_line)
        return

    width, height = image.size

    # Correct bug in the streetview library to remove empty parts in the image array
    if width>TYPICAL_WIDTH or height>TYPICAL_HEIGHT:
        if width>TYPICAL_WIDTH:
            # Verify that the horizontal empty space has the correct size
            empty_width = image.crop(box=(TYPICAL_WIDTH, 0, width, height))
            max_sum = np.sum(np.array(empty_width))

            if max_sum!=0:
                print(f'Image {image_name} has an abnormal width ({width}), a txt file with its name will be stored instead.')
                with open(os.path.join(SAVE_PATH, image_name + '.txt'), 'w') as tx:
                    tx.write(panorama_file_line)
                return
            else:
                image = image.crop(box=(0, 0, TYPICAL_WIDTH, height))
                width = TYPICAL_WIDTH

        if height>TYPICAL_HEIGHT:
            # Verify that the vertical empty space has the correct size
            empty_height = image.crop(box=(0, TYPICAL_HEIGHT, width, height))
            max_sum = np.sum(np.array(empty_height))
            
            if max_sum!=0:
                print(f'Image {image_name} has an abnormal height ({height}), a txt file with its name will be stored instead.')
                with open(os.path.join(SAVE_PATH, image_name + '.txt'), 'w') as tx:
                    tx.write(panorama_file_line)
                return
            else:
                image = image.crop(box=(0, 0, width, TYPICAL_HEIGHT))

    width, height = image.size
    if args.dont_preprocess:
        image.save(os.path.join(SAVE_PATH, image_name + '.jpg'), "jpeg")
    else:
        # Remove 1328 pixels from the top and the bottom to remove artifacts from panoramic image formation
        assert height== TYPICAL_HEIGHT, height
        image = image.crop(box=(0, CROP_HEIGHT_INDEX, width, POST_HEIGHT))
        image.save(os.path.join(SAVE_PATH, image_name + '.jpg'), "jpeg")

with open(args.ids_file_path, 'r') as f:
    panoramas_list = f.readlines()

if args.percentage<1:
    panoramas_list = panoramas_list[:round(len(panoramas_list)*args.percentage)]

if args.num_processes>1:
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        _ = list(tqdm(pool.imap_unordered(download_panoramas, panoramas_list)))
else:
    for pano_file_line in tqdm(panoramas_list):
        download_panoramas(pano_file_line)
