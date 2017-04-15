import ast
import numpy as np
import re

class combine(object):
    #Want to be able to combine the labels from multiple different partitions into one overall classification
    def __init__(self):
        try:
            info_file_path = "./output/deploy_output/info.txt"
            info_file = open(info_file_path)
        except:
            print("Problem opening info file " + info_file_path)
        #self.batch_size = info_file.readline()
        self.num_partitions_per_image = info_file.readline()
        self.image_dim = ast.literal_eval(info_file.readline())
        #partition_indices_string = info_file.readline()
        #self.partition_indices = partition_indices_string.split()
        self.im_files = [f.split() for f in info_file.readlines()]
        self.normalising_array = self.calc_normalising_array()
        info_file.close()

    def calc_normalising_array(self):
        """
        Calculate the normalising value for each pixel in the array

        Returns:
            next_level_weights (numpy.array): A 2D numpy array whose entries denote the normalising value of each output pixel

        """
        image = np.zeros(self.image_dim)

        if __name__ == '__main__':
            if __name__ == '__main__':
                if __name__ == '__main__':
                    for j in range(len(self.im_files[0])):
                        image_pattern = r"./output/deploy_output/(?P<imag_num>\d+)_image_x(?P<x_offset>\d+)_y(?P<x_offset>\d+).png"
                        r = re.findall(image_pattern, self.im_files[0][j])
                        _, x_off, y_off = int(r[0]), int(r[1]), int(r[2])
                        #Need to add numpy array to a smaller numpy array, with an offset given by the x_of and y_off. Note we want to apply some
                        #function to each pixel, we do no necessarily think that this relationship is linear


    def calculate_labels(self):
        #Looping over each image
        for i in range(len(self.im_files)):
            image = np.zeros(self.image_dim)
            for j in range(len(self.im_files[i])):
                image_pattern = r"./output/deploy_output/(?P<imag_num>\d+)_image_x(?P<x_offset>\d+)_y(?P<x_offset>\d+).png"
                r = re.findall(image_pattern, self.im_files[i][j])
                im_num, x_off, y_off = int(r[0]), int(r[1]), int(r[2])
