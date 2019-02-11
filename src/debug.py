#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys
import cv2

import matplotlib.pyplot as plt
import random

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.visualization import draw_annotations, draw_boxes
from keras_retinanet.utils.anchors import anchors_for_shape, compute_gt_annotations, AnchorParameters
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters


def create_generator(args):
    """ Create the data generators.

    Args:
        args: parseargs arguments object.
    """
    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )

    generator = CSVGenerator(
        args.annotations,
        args.classes,
        transform_generator=transform_generator,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side,
        config=args.config
    )

    return generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Debug script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path',  help='Path to dataset directory (ie. /tmp/COCO).')
    coco_parser.add_argument('--coco-set', help='Name of the set to show (defaults to val2017).', default='val2017')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--pascal-set',  help='Name of the set to show (defaults to test).', default='test')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')
    kitti_parser.add_argument('subset', help='Argument for loading a subset from train/val.')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('subset', help='Argument for loading a subset from train/validation/test.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes',     help='Path to a CSV file containing class label mapping.')

    parser.add_argument('-l', '--loop', help='Loop forever, even if the dataset is exhausted.', action='store_true')
    parser.add_argument('--no-resize', help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--anchors', help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--annotations', help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')

    return parser.parse_args(args)


def save_anchor_params_in_tmp_file(params, filename):
    params_file = open(filename, "w")

    params_file.write('[anchor_parameters]\n');
    for key, value in params.items():
        params_file.write(key + ' = ' + ' '.join(map(str, value)) + '\n')

    params_file.close()

def get_args_object(user_anchor_params, annotations_file, classes_file):
    # In order to not modify any keras_retinanet functions, we'll store the passed params to a file and will pass it as argument to the keras_retinanet functions
    tmp_achor_params_file = '/tmp/anchor_params.tmp'
    save_anchor_params_in_tmp_file(user_anchor_params, tmp_achor_params_file)

    args = parse_args(('--anchors --config ' + tmp_achor_params_file + ' csv ' + annotations_file + ' ' + classes_file + '').split(' '))

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)


    return args

def should_skip_image(generator, i):
    return len(generator.load_annotations(i)['labels']) == 0

def get_anchor_indicies_for_image(args, pyramid_levels, generator, i):
    # load the data
    image       = generator.load_image(i)
    annotations = generator.load_annotations(i)

    # apply random transformations
    if args.random_transform:
        image, annotations = generator.random_transform_group_entry(image, annotations)

    # resize the image and annotations
    if args.resize:
        image, image_scale = generator.resize_image(image)
        annotations['bboxes'] *= image_scale

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)
    
    anchors = anchors_for_shape(image.shape, pyramid_levels=pyramid_levels, anchor_params=anchor_params)
    positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])

    return positive_indices, max_indices, image, anchors, annotations


def show_images_with_anchors(user_anchor_params, pyramid_levels, annotations_file, classes_file, limit=10):
    args = get_args_object(user_anchor_params, annotations_file, classes_file)

    # create the generator
    generator = create_generator(args)

    # display images, one at a time
    random.seed(40)
    for i in random.sample(range(0, generator.size()), limit):
        if should_skip_image(generator, i):
            continue

        positive_indices, max_indices, image, anchors, annotations = get_anchor_indicies_for_image(args, pyramid_levels, generator, i)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 10))

        img_1 = image.copy()
        print ('Image', generator.image_path(i))

        # draw anchors on the image
        if args.anchors:
            draw_boxes(img_1, anchors[positive_indices], (255, 255, 0), thickness=1)

        # draw annotations on the image
        if args.annotations:
            # draw annotations in red
            draw_annotations(img_1, annotations, color=(255, 0, 0), label_to_name=generator.label_to_name)

            # draw regressed anchors in green to override most red annotations
            # result is that annotations without anchors are red, with anchors are green
            draw_boxes(img_1, annotations['bboxes'][max_indices[positive_indices], :], (0, 255, 0))

        ax1.imshow(image)
        ax1.set_title('Image')
        ax2.set_title('Image with derived bounding box')
        ax2.imshow(img_1)

        plt.show()


def count_ignored_ships(user_anchor_params, pyramid_levels, annotations_file, classes_file, limit=0):
    args = get_args_object(user_anchor_params, annotations_file, classes_file)

    # create the generator
    generator = create_generator(args)

    ignored_ships = 0
    total_ships = 0
    for i in range(0, generator.size()):
        if should_skip_image(generator, i):
            continue

        positive_indices, max_indices, _, _, annotations = get_anchor_indicies_for_image(args, pyramid_levels, generator, i)

        ignored_bboxes={}
        for bbox in annotations['bboxes']:
            ignored_bboxes[(bbox[0], bbox[1], bbox[2], bbox[3])] = True

        for bbox in annotations['bboxes'][max_indices[positive_indices], :]:
            if (bbox[0], bbox[1], bbox[2], bbox[3]) in ignored_bboxes:
                del ignored_bboxes[(bbox[0], bbox[1], bbox[2], bbox[3])]

        if len(ignored_bboxes):
            ignored_ships += len(ignored_bboxes)

        total_ships += len(annotations['bboxes'])

        if limit > 0 and i >= limit:
            break

    print(ignored_ships, ' ships ignored out of ', total_ships)
