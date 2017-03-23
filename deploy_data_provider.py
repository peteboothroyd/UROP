import os
import caffe
import numpy as np
import json

import skimage.io as imio
import skimage.transform as imtf
import random

from upconvolveinformation import UpConvolve

join = os.path.join

class DeployDataLayer(caffe.Layer):
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
        self.image_list = params['image_list']
        self.height = params['height']
        self.width = params['width']

        self.test_sample = 0
        self.num_caffe_requested_iter = 0

        with open(self.image_list, 'r') as fid:
            self.im_files = [join(self.image_path,f.strip()) for f in fid.readlines()]

        self.n_files = len(self.im_files)

        try:
            impath = join(self.image_path, self.im_files[0])
            im = imio.imread(impath)
            image_height, image_width, _ = im.shape
            self.image_width, self.image_height = image_width, image_height
            print("Set up DeployDataLayer: ")
            print("batch size = " + str(self.batch_size))
            print("image path = " + str(self.image_path))
            print("image shape = " + str([self.image_width, self.image_height]))
        except:
            print("Problem loading image with path: " + impath)

        self.partition_indices = self.partition(self.stride, self.kernel_size, [self.width, self.height], self.num_conv_levels)
        self.num_partitions_per_image = len(self.partition_indices)

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, self.channels, self.height, self.width)

    def forward(self, bottom, top):
        top[0].data[...] = 0

        try:
            filepath = "./output/deploy_output/info.txt"
            file = open(filepath, "w+")
            file.write(str(self.batch_size) + "\n")
            file.write(str(len(self.partition_indices)) + "\n")
            file.write(str(self.partition_indices) + "\n")
        except IOError:
            print("Problem opening file: " + filepath)

        for i in range(self.batch_size):
            while True:
                while True:
                    r = random.randrange(0, self.n_files)
                    try:
                        impath = join(self.image_path, self.im_files[r])
                        im = imio.imread(impath)
                        #print("im_shape: " + str(im.shape))
                    except:
                        print("Problem loading image with path: " + impath)
                    else:
                        break

                for j in range(len(self.partition_indices)):
                    startx, stopx = self.partition_indices[j][0][0], self.partition_indices[j][0][1]
                    starty, stopy = self.partition_indices[j][1][0], self.partition_indices[j][1][1]
                    #print("Startx: " + str(startx) + ". Stopx: " + str(stopx) +". Starty: " + str(starty) +". Stopy: " + str(stopy))

                    cropped_im = im[starty:stopy, startx:stopx, :]
                    #print("im_shape after cropping: " + str(cropped_im.shape))

                    image_num = int(self.test_sample / self.num_partitions_per_image)
                    output_cropped_im_path = "./output/deploy_output/{0}_image_x{1}_y{2}.png".format(image_num, startx, starty)
                    #print("Output image path: " + output_cropped_im_path)
                    imio.imsave(output_cropped_im_path, cropped_im)

                    #file.write(output_cropped_im_path + " ")

                    cropped_im = cropped_im.transpose(2, 0, 1).astype(np.float32)

                    top[0].data[i, ...] = cropped_im
                    self.test_sample += 1

                file.write("\n")
                break

        file.close()

    def partition_indices(self, step, partition_size):
            """
            Takes an input array size and partitions this up into possibly overlapping windows. Note the window size is
            found from the self.height and self.width properties.
            Args:
                step ([int, int]): A 2D array denoting the step size in the x, y directions
                partition_size ([int, int]): A 2D array denoting the partition window size in the x, y directions

            Returns:
                partition_indices ([[int, int], [int, int]]): A list of start and end indices for each partition
            """

            partition_indices = []
            current_x, current_y = 0, 0
            while current_y < self.image_height:
                while current_x < self.image_width:
                    if current_x + partition_size[0] < self.image_width:
                        startx, stopx = current_x, current_x + partition_size[0]
                    else:
                        startx, stopx = self.image_width - partition_size[0], self.image_width
                        current_x = self.image_width
                    if current_y + partition_size[1] < self.image_height:
                        starty, stopy = current_y, current_y + partition_size[1]
                    else:
                        starty, stopy = self.image_height - partition_size[1], self.image_height

                    partition_indices.append([[startx, stopx], [starty, stopy]])
                    current_x += step[0]

                current_x = 0
                current_y += step[1]
            #print("Partition indices:" + str(partition_indices))
            return partition_indices

    def partition(self, stride, kernel_size, partition_size, num_conv_layers):
        upconv = UpConvolve(stride, kernel_size, partition_size, num_conv_layers)
        step = upconv.find_central_window_dimensions()
        return self.partition_indices(step, partition_size)

