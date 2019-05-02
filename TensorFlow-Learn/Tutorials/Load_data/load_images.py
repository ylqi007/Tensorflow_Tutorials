from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import pathlib
import random
import IPython.display as display

tf.enable_eager_execution()

print(tf.VERSION)   # 1.11.0-dev20180917


# 1. Download and inspect the dataset
## 1.1 Retrieve the images
def _download_dataset():
    data_root = tf.keras.utils.get_file('flower_photos',
                                        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                        untar=True)
    data_root = pathlib.Path(data_root)
    print('data_root: ', data_root)
    for item in data_root.iterdir():
        print("##: ", item)
    _all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in _all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    print(image_count)
    for item in all_image_paths[:10]:
        print(item)

    return data_root, all_image_paths


## 1.2 Inspect the images
def _inspect_the_images(data_root):
    attributions = (data_root/"LICENSE.txt").read_text(encoding='utf8').splitlines()[4:]
    attributions = [line.split('CC-BY') for line in attributions]
    attributions = dict(attributions)
    return attributions


def _caption_image(data_root, all_image_paths, attributions, image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + " - ".join(attributions[str(image_rel)].split(' - ')[:-1])


## 1.3 Determine the label for each image

## 1.4 Load and format the images

# 2. Build a `tf.data.Dataset`
## 2.1 A dataset of images

## 2.2 A dataset of (image, label) pairs

## 2.3 Basic methods for training

## 2.4 Pipe the dataset to a model

# 3. Performance
## 3.1 Cache

## 3.2 TFRecord File


if __name__ == '__main__':
    data_root, all_image_paths = _download_dataset()
    attributions = _inspect_the_images(data_root)
    for n in range(3):
        image_p = random.choice(all_image_paths)
        display.display(display.Image(image_p))