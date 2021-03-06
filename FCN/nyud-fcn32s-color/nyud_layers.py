import caffe

import numpy as np
from PIL import Image
#import scipy.io
import os

import random

join = os.path.join

class NYUDSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from NYUDv2
    one-at-a-time while reshaping the net to preserve dimensions.

    The labels follow the 40 class task defined by

        S. Gupta, R. Girshick, p. Arbelaez, and J. Malik. Learning rich features
        from RGB-D images for object detection and segmentation. ECCV 2014.

    with 0 as the void label and 1-40 the classes.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - split: train / val / test
        - tops: list of tops to output from {color, depth, hha, label}
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for NYUDv2 semantic segmentation.

        example: params = dict(nyud_dir="/path/to/NYUDVOC2011", split="val",
                               tops=['color', 'hha', 'label'])
        """
        # config
        params = eval(self.param_str)
        self.split = params['split']
        self.tops = params['tops']

        self.image_path = params['image_path']
        self.label_path = params['label_path']
        self.image_list = params['image_list']
        self.label_list = params['label_list']

        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # store top data for reshape + forward
        self.data = {}

        # means
        #TODO: Figure out mean/ just minus np.mean
        '''
        self.mean_bgr = np.array((116.190, 97.203, 92.318), dtype=np.float32)
        self.mean_hha = np.array((132.431, 94.076, 118.477), dtype=np.float32)
        self.mean_logd = np.array((7.844,), dtype=np.float32)
        '''

        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        ### LOADING IMAGE AND LABEL LOCATIONS ###
        with open(self.image_list, 'r') as fid:
            self.im_files = [join(self.image_path, f.strip()) for f in fid.readlines()]

        with open(self.label_list, 'r') as fid:
            self.gt_files = [join(self.label_path, f.strip()) for f in fid.readlines()]

        self.n_files = len(self.im_files)

        assert self.n_files == len(self.gt_files), 'Number of images and labels differ!'

        '''
        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.nyud_dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        '''
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, self.n_files-1)

    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, self.idx)
            top[i].reshape(1, *self.data[t].shape)

    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

        # pick next input
        if self.random:
            self.idx = random.randint(0, self.n_files-1)
        else:
            self.idx += 1 % self.n_files
    def backward(self, top, propagate_down, bottom):
        pass

    def load(self, top, idx):
        if top == 'color':
            return self.load_image(idx)
        elif top == 'label':
            return self.load_label(idx)
        else:
            raise Exception("Unknown output type: {}".format(top))

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open(self.im_files[idx])
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.mean(in_, axis=(0,1))
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 0-39 and void is 255 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
        label = Image.open(self.gt_files[idx])
        in_ = np.array(label, dtype=np.uint8)
        #label = scipy.io.loadmat('{}/segmentation/img_{}.mat'.format(self.nyud_dir, idx))['segmentation'].astype(np.uint8)
        #in_ -= 1  # rotate labels, is this necessary?
        print("loading label")
        in_ = np.sum(in_, axis=2)
        in_[in_ != 0] = 1
        in_ = in_[np.newaxis, ...]
        #mat_in_ = np.matrix(in_)
        return in_
