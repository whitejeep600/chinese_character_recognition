import json
import os
import re

from PIL import Image
from tqdm import tqdm

from preprocessing import preprocess


def read_images():
    all_images = []
    progress = tqdm(total=len(os.listdir('data')), desc="Preprocessed characters")
    for dirname in os.listdir('data'):
        new_images = []
        this_directory_character = None
        for filename in os.listdir(os.path.join('data', dirname)):
            found_character = re.search(r'(.)_[0-9]*.png', filename).group(1)
            if this_directory_character is None:
                this_directory_character = found_character
            else:
                assert this_directory_character == found_character
            with Image.open(os.path.join('data', dirname, filename)) as image:
                new_images.append({'image': preprocess(image),
                                   'label': this_directory_character})
        all_images += new_images
        progress.update(1)
    return all_images


def read_label_mapping():
    with open('label_to_int.json', 'r') as file:
        return json.loads(file.read())


def read_all_characters():
    with open('all_characters.json', 'r') as file:
        return json.loads(file.read())
