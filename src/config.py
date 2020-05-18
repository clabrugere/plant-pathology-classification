import os
from pathlib2 import Path


DIR_ROOT = Path(os.path.abspath(__file__)).parents[1]
DIR_DATA = os.path.join(DIR_ROOT, 'data/')
DIR_IMAGES = os.path.join(DIR_DATA, 'images/')
DIR_MODELS = os.path.join(DIR_ROOT, 'models/')

METADATA_TRAIN = os.path.join(DIR_DATA, 'train.csv')
METADATA_TEST = os.path.join(DIR_DATA, 'test.csv')


def get_image_filename(name):
    return os.path.join(DIR_IMAGES, f'{name}.jpg')

def get_model_filename(name):
    return os.path.join(DIR_MODELS, f'{name}.pth')