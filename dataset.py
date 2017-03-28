from __future__ import print_function

import os
import cv2
import numpy as np
import random
import glob

from keras.preprocessing.image import transform_matrix_offset_center, apply_transform
from scipy.ndimage.filters import gaussian_filter
from keras import backend as K

class Dataset():
    def __init__(self, args_dict):
        self.wd = args_dict['wd']
        self.ht = args_dict['ht']
        self.train_dir = args_dict['train_dir']
        self.test_dir = args_dict['test_dir']

        self.training_ignore_list_file = os.path.join('resources', 'training_ignore_list.txt')

        # Prepare sketch lists
        self.full_train_sketch_list = self._get_sketch_list('train')
        train_split = 100 - self.val_split
        # Divide between training and validation set
        num_sketches_train = int((len(self.full_train_sketch_list) * train_split) / 100)
        self.train_sketch_list = self.full_train_sketch_list[:num_sketches_train]
        self.val_sketch_list = self.full_train_sketch_list[num_sketches_train:]
        self.test_sketch_list = self._get_sketch_list('test')
        self.limited_train_sketch_list = self.full_train_sketch_list

        self._print_dataset_config()
        
    def _print_dataset_config(self):
        return

    def get_input_dim(self):
        if K.image_dim_ordering() == 'tf':
            return (self.ht, self.wd, 1)
        else:
            return (1, self.ht, self.wd)

    def _get_sketch(self, sketch_path):
        sketch = cv2.imread(sketch_path)
        if sketch is not None:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
            sketch = cv2.resize(sketch, (self.wd, self.ht))
            sketch = 255 - sketch # Make background zero by inverting
            sketch = sketch.astype('float32')

            # Some sketches a greyish background. Eliminate it.
            # sketch[sketch < 30] = 0

            # Bring the image data to -1 to +1 range
            sketch = (sketch * 2)/255. - 1
            return sketch
        else:
            print('Unable to open ', sketch_path, ' Skipping.')
            return None

    def prep_training(self, train_args):
        self.train_mode = train_args['train_mode']

        self.val_split = train_args['val_split']
        self.use_augmentation = train_args['use_augmentation']
        self.num_additional_sketches = train_args['num_additional_sketches']
        self.height_shift_range = train_args['height_shift_range']
        self.width_shift_range = train_args['width_shift_range']
        self.rotation_range = train_args['rotation_range']
        self.shear_range = train_args['shear_range']
        self.zoom_range = train_args['zoom_range']
        self.fill_mode = train_args['fill_mode']
        self.cval = train_args['cval']

        if not self.use_augmentation:
            self.num_additional_sketches = 0

        self._print_train_config()
        
    def _print_train_config(self):
        return

    def prep_test(self, test_args):
        return

    def _get_sketch_list(self, phase):
        if phase == 'train':
            sketch_list = os.listdir(self.train_dir)
            if len(sketch_list) < 1:
                print('Found only {0:d} sketches in the sketch directory: {1:s}'.\
                    format(len(sketch_list), self.train_dir),
                    'What are you trying to do? Aborting for now.')
                exit(0)

            # Get list of sketches to ignore
            ignore_list = open(self.training_ignore_list_file, 'r').read().splitlines()
            ignore_list = [i for i in ignore_list if i.isspace() is False and i.startswith('#') is False]
            # Remove the sketches that are present in ignore_list
            sketch_list = [i for i in sketch_list if i not in ignore_list]
        else: # test
            sketch_list = os.listdir(self.test_dir)
            if len(sketch_list) < 1:
                print('Found only {0:d} sketches in the sketch directory: {1:s}'.\
                    format(len(sketch_list), self.test_dir),
                    'What are you trying to do? Aborting for now.')
                exit(0)

        # Shuffle the list
        random.shuffle(sketch_list)
        return sketch_list

    def get_batch(self, batch_size, sketch_set, limit_search_space=False):
        if sketch_set == 'train_set':
            sketch_dir = self.train_dir
            sketch_list = self.train_sketch_list
        elif sketch_set == 'val_set':
            sketch_dir = self.train_dir
            sketch_list = self.test_sketch_list
        elif sketch_set == 'test_set':
            sketch_dir = self.train_dir
            sketch_list = self.test_sketch_list
        elif sketch_set == 'full_train_set':
            sketch_dir = self.train_dir
            sketch_list = self.full_train_sketch_list
        elif sketch_set == 'limited_train_set':
            sketch_dir = self.train_dir
            sketch_list = self.limited_train_sketch_list
        else:
            print('Error: Weird "sketch_set" arg to get_batch():{0:s}'.format(sketch_set))
            exit(0)

        ht = self.ht
        wd = self.wd

        if K.image_dim_ordering() == 'tf':
            img_shape = (ht, wd, 1)
        else:
            img_shape = (1, ht, wd)
        X = np.zeros((batch_size,) + img_shape)

        src_idx = 0
        dst_idx = 0
        while True:
            sketch_name = sketch_list[src_idx]
            X[dst_idx] = self._get_sketch(os.path.join(sketch_dir, sketch_name)).reshape(img_shape)

            if self.use_augmentation is True:
                X[dst_idx] = self._apply_affine_distortion(X[dst_idx])

            src_idx += 1
            dst_idx += 1
            if src_idx >= len(sketch_list):
                src_idx = 0
            if dst_idx >= batch_size:
                dst_idx = 0
                yield X, X

    # Function definition taken from keras source code
    def _apply_affine_distortion(self, x):
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = 0
        img_col_axis = 1
        img_channel_axis = 0

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                                translation_matrix),
                                         shear_matrix),
                                  zoom_matrix)

        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis,
                            fill_mode=self.fill_mode, cval=self.cval)

        return x

    '''
    Function to visualize how the sketches look after making them
    zero mean and unit variance
    '''
    def _dump_sketces(self, sketch_dir, sketch_list, num_sketches_to_dump):
        if not os.path.exists(sketch_dir):
            print('The sketch directory {0:s} does not exist'.format(sketch_dir))
            return

        wd = self.wd
        ht = self.ht

        count = 0
        for i in range(num_sketches_to_dump):
            sketch = self._get_sketch(os.path.join(sketch_dir, sketch_list[i]))
            # This sketch is zero-mean and unit-variance. In order to write
            # it on disk, we first need to bring it 0-255 range
            sketch = (1 + sketch) * 255/2.
            sketch = np.clip(sketch, 0, 255)
            cv2.imwrite( os.path.join('temp', 'Aug_' + sketch_list[i]), sketch)
            if count >= num_sketches_to_dump:
                break

    def validate_dataset(self, batch_size):
        num_sketches_to_dump = 10
        print('Dumping sketches for debugging....')
        if 0:
            self._dump_sketces(self.train_dir, self.train_sketch_list, num_sketches_to_dump)
            self._dump_sketces(self.train_dir, self.val_sketch_list, num_sketches_to_dump)
            self._dump_sketces(self.test_dir, self.test_sketch_list, num_sketches_to_dump)
            exit(0) # Usually I want to exit after dumping sketches


