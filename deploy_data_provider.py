import os
import caffe
import numpy as np
import json

import skimage.io as imio
import skimage.transform as imtf
import random
import IO

from upconvolveinformation import UpConvolve

join = os.path.join

class DeployDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(top) != 3:
            raise ValueError('DataLayer should have 3 outputs for deployment (image, label, offset)')

        random.seed()

        self.output_path = "./output/deploy_output/"

        # NOTE(fixme, dirty)
        self.channels = 3
        self.stride = [2, 2]
        self.kernel_size = [4, 4]
        self.num_conv_levels = 5 #This is the number of up/down convolve layers in the architecture which determines the partition dimensions

        params = json.loads(self.param_str)

        try:
            self.batch_size = params['batch_size']
            self.image_path = params['image_path']
            self.label_path = params['label_path']
            self.image_list = params['image_list']
            self.label_list = params['label_list']
            self.partition_height = params['height']
            self.partition_width = params['width']
        except KeyError:
            print("Problem getting parameters from the data layer. Check that all the required parameters are supplied")

        self.test_sample = 0

        with open(self.image_list, 'r') as fid:
            self.im_files = [join(self.image_path, f.strip()) for f in fid.readlines()]

        with open(self.label_list, 'r') as fid:
            self.gt_files = [join(self.label_path, f.strip()) for f in fid.readlines()]

        self.n_files = len(self.im_files)

        assert self.n_files == len(self.gt_files), 'Number of images and labels differ!'

        while True:
            r = random.randrange(0, self.n_files)
            try:
                impath = self.im_files[r]
                self.image_height, self.image_width, _ = IO.find_partition_dim(impath)
            except Exception as e:
                print(e)
                print("Problem loading image with path: " + impath)
            else:
                break

        self.upConv = UpConvolve(self.stride, self.kernel_size, [self.partition_width, self.partition_height], [self.image_width, self.image_height], self.num_conv_levels)
        self.partition_indices = self.upConv.partition()
        self.num_partitions_per_image = len(self.partition_indices)

        IO.create_info_file(self.output_path + "info.txt", len(self.partition_indices), [self.image_width, self.image_height], self.stride, self.kernel_size, self.num_conv_levels, self.partition_indices)

        print("Set up DeployDataLayer.")
        print("image shape(height, width) = " + str([self.image_height, self.image_width]))

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, self.channels, self.partition_height, self.partition_width)
        top[1].reshape(self.batch_size, 1, self.partition_height, self.partition_width)
        top[2].reshape(self.batch_size, 2)

    def forward(self, bottom, top):
        top[0].data[...] = 0
        top[1].data[...] = 0

        for i in range(self.batch_size): #Batch size should equal the number of partitions which we are splitting the image up into
            while True:
                r = random.randrange(0, self.n_files)
                try:
                    impath = self.im_files[r]
                    labelpath = self.gt_files[r]
                    im = imio.imread(impath)
                    label = imio.imread(labelpath).astype(np.int32)
                    image_num = int(self.test_sample / self.num_partitions_per_image)
                    if image_num <= 1:
                        imio.imsave("./output/deploy_output/target_image{}.png".format(image_num), im)
                        imio.imsave("./output/deploy_output/target_label{}.png".format(image_num), label)
                except Exception as e:
                    print(e)
                    print("Deploy Forward: IOError.")
                else:
                    break

            startx, stopx = self.partition_indices[i][0][0], self.partition_indices[i][0][1]
            starty, stopy = self.partition_indices[i][1][0], self.partition_indices[i][1][1]
            offset = np.array([startx, starty])

            cropped_im = im[starty:stopy, startx:stopx, :]
            cropped_label = label[starty:stopy, startx:stopx, :]

            cropped_im = cropped_im.transpose(2, 0, 1).astype(np.float32)
            cropped_label = np.sum(cropped_label, axis=2)
            cropped_label[cropped_label != 0] = 1

            top[0].data[i, ...] = cropped_im
            top[1].data[i, ...] = cropped_label
            top[2].data[i, ...] = offset
            self.test_sample += 1

            ##DEBUG##
            output_cropped_im_path = self.output_path + "{0}_x{1}_y{2}_image.png".format(image_num, startx, starty)
            output_cropped_label_path = self.output_path + "{0}_x{1}_y{2}_label.png".format(image_num, startx, starty)
            imio.imsave(output_cropped_im_path, cropped_im)
            imio.imsave(output_cropped_label_path, cropped_label)
            ##DEBUG##
