import os
import caffe
import numpy as np
import json

import skimage.io as imio
import skimage.transform as imtf
import random

from upconvolveinformation import UpConvolve

join = os.path.join

class DataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(top) != 1:
            raise ValueError('DataLayer should have 1 outputs for deployment.')

        random.seed()

        # NOTE(fixme, dirty)
        self.channels = 3
        self.stride = [2, 2]
        self.kernel_size = [4, 4]
        self.num_conv_levels = 5

        params = json.loads(self.param_str)
        self.batch_size = params['batch_size']
        self.image_path = params['image_path']
        self.label_path = params['label_path']
        self.image_list = params['image_list']
        self.label_list = params['label_list']
        self.height = params['height']
        self.width = params['width']

        with open(self.image_list, 'r') as fid:
            self.im_files = [join(self.image_path,f.strip()) for f in fid.readlines()]

        self.n_files = len(self.im_files)

        try:
            impath = join(self.image_path, self.im_files[0])
            im = imio.imread(impath)
            image_height, image_width, _ = im.shape
            self.image_shape = [image_width, image_height]
        except:
            print("Problem loading image with path: " + impath)

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, self.channels, self.height, self.width)

    def forward(self, bottom, top):
        top[0].data[...] = 0

        partition_indices = self.partition(self.stride, self.kernel_size, self.image_shape, self.num_conv_levels)

        for i in range(self.batch_size):
            while True:
                while True:
                    r = random.randrange(0, self.n_files)
                    try:
                        impath = join(self.image_path, self.im_files[r])
                        im = imio.imread(impath)
                    except:
                        print("Problem loading image with path: " + impath)
                    else:
                        break

                for j in range(len(partition_indices)):
                    startx, stopx = partition_indices[j][1][0], partition_indices[j][1][1]
                    starty, stopy = partition_indices[j][0][0], partition_indices[j][0][1]
                    im = im[startx:stopx, starty:stopy, :]

                    imio.imsave("./output/deploy_output/{0}_image_x{1}_y{2}.png".format(j, startx, starty), im)

                    im = im.transpose(2, 0, 1).astype(np.float32)

                    top[0].data[i, ...] = im

                break

    def backward(self, bottom, top):
        raise NotImplemented

    def partition_indices(self, step, partition_size):
        """
        Takes an input array size and partitions this up into possibly overlapping windows
        Args:
            step ([int, int]): A 2D array denoting the step size in the x, y directions
            size ([int, int]): A 2D array denoting the partition window size in the x, y directions

        Returns:
            partition_indices ([[int, int], [int, int]]): A list of start and end indices for each partition
        """

        partition_indices = []
        current_x, current_y = 0, 0
        while current_y < self.height:
            while current_x < self.width:
                if current_x + step[0] < self.width:
                    startx, stopx = current_x, current_x + partition_size[0]
                else:
                    startx, stopx = self.width - partition_size[0], self.width
                if current_y + step[1] < self.height:
                    starty, stopy = current_y, current_y + partition_size[1]
                else:
                    starty, stopy = self.height - partition_size[1], self.height

                partition_indices.append([[startx, stopx], [starty, stopy]])
                current_x += step[0]

            current_x = 0
            current_y += step[1]

        return partition_indices

    def partition(self, stride, kernel_size, window_size, num_conv_layers):
        upconv = UpConvolve(stride, kernel_size, window_size, num_conv_layers)
        step = upconv.find_central_window_dimensions()
        return self.partition_indices(step, window_size)

def test():
    pass

if __name__ == "__main__":
  test()
