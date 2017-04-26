import numpy as np
import scipy.misc

class UpConvolve(object):
    """Class to calculate the relative information content of different pixels in an upconvolved image.
        In a deep learning architecture such as the U-Net or more recent delta-net up convolving from the lowest level
        inevitably means that padding has to be applied. Edge pixels in the final segmentation are generated from
        automatically generated padded pixels which leads to relatively worse performance towards the edge of the output.
        This is aimed at 2D input data.
    """
    def __init__(self, stride, kernel_size, partition_size, image_size, n_conv_layers):
        """
        Args:
            stride ([int, int]):        List of ints with 2 elements; [x-stride, y-stride]. Note normally both these values will be equal.
            kernel_size ([int, int]):   The kernel size for downsampling [x-size, y-size]. Note normally both these values will be equal.
            partition_size ([int, int]):The partition size which we are considering (eg. 94x94px).
            image_size ([int, int]):    The size of the overall image which we are partitioning up. [im-x-size, im-y-size]
            n_conv_layers (int):        The number of up/down convolution layers in the architecture.

        """
        self.stride_x = stride[0]
        self.stride_y = stride[1]

        self.kernel_size_x = kernel_size[0]
        self.kernel_size_y = kernel_size[1]

        self.partition_size_x = partition_size[0]
        self.partition_size_y = partition_size[1]

        self.im_size_x = image_size[0]
        self.im_size_y = image_size[1]

        self.kernel = np.ones((self.kernel_size_y, self.kernel_size_x))
        self.conv_layers = n_conv_layers

    def generate_weights(self):
        """
        Produces a 2D array for the given kernel size, stride and number of upconvolution layers

        Returns:
            information_heatmap (numpy.array): A 2D numpy array whose entries denote the 'information' of each output pixel

        """
        x, y = self.calc_lowest_level_dim()

        current_arr = np.ones((y, x))

        for i in range(self.conv_layers):
            current_arr = self.gen_next_level_weights(current_arr)

        return current_arr

    def gen_next_level_weights(self, previous_level_weights):
        """
        Method which generates the next level of 'information' weights based upon the previous level
        Args:
            previous_level_weights (numpy.array): A 2D numpy array whose entries denote the 'information' of each input pixel

        Returns:
            next_level_weights (numpy.array): A 2D numpy array whose entries denote the 'information' of each output pixel

        """
        prev_num_rows, prev_num_cols = previous_level_weights.shape
        new_num_cols = prev_num_cols * self.stride_x + (self.kernel_size_x - self.stride_x)
        new_num_rows = prev_num_rows * self.stride_y + (self.kernel_size_y - self.stride_y)

        #Want to dilate this layer, use row-major format
        next_level = np.zeros((new_num_rows, new_num_cols))

        for row in range(prev_num_rows):
            for col in range(prev_num_cols):
                for i in range(self.kernel_size_x):
                    for j in range(self.kernel_size_y):
                        next_level[row * self.stride_y + i][col * self.stride_x + j] += previous_level_weights[row][col]

        return next_level

    def write_to_im(self, array, filename):
        """
        Writes a numpy array to an image file to allow easy viewing of the intensities
        Args:
            array (numpy.array): A 2D numpy array whose entries denote the 'information' of each input pixel
        """
        scipy.misc.imsave(filename, array)

    def calc_lowest_level_dim(self):
        """
        Calculates the dimensions of the lowest level in the network using the input image dimensions, stride, kernel size and number of
        up/down convolutional layers
        Returns:
            lowest_layer_dims ([int, int]): The size of the 2D image at the lowest level of the network.
        """
        current_x = self.partition_size_x
        current_y = self.partition_size_y

        step_x = self.kernel_size_x - self.stride_x
        step_y = self.kernel_size_y - self.stride_y

        for _ in range(self.conv_layers):
            current_x = (current_x - step_x) / self.stride_x
            current_y = (current_y - step_y) / self.stride_y

        return int(current_x), int(current_y)

    def find_central_window_dimensions(self):
        """
        In the centre of the weight matrix will be a rectangle with maximum valued entries, the dimensions of this rectangle will be
        found and returned

        Returns:
            dims ([int, int]): The size of the 2D rectangle in the centre of the weight matrix with maximal value ([x, y])
        """
        weight_matrix = self.generate_weights()

        height, width = weight_matrix.shape
        centre_x = int(width / 2)
        centre_y = int(height / 2)
        max_val = weight_matrix[centre_y][centre_x]

        startx, stopx = 0, 0

        for i in range(width):
            if weight_matrix[centre_y][i] == max_val:
                startx = i
                for j in range(i, width):
                    if weight_matrix[centre_y][j] < max_val:
                        stopx = j
                        break
                break

        for i in range(height):
            if weight_matrix[i][centre_y] == max_val:
                starty = i
                for j in range(i, width):
                    if weight_matrix[centre_y][j] < max_val:
                        stopy = j
                        break
                break

        return [stopx - startx, stopy - starty]

    def partition(self):
        """
            Takes an input array size and partitions this up into possibly overlapping windows.

            Returns:
                partition_indices ([[int, int], [int, int]]): A list of start and end indices for each partition
            """
        step = self.find_central_window_dimensions()

        partition_indices = []

        current_x, current_y = 0, 0
        image_width, image_height = self.im_size_x, self.im_size_y
        partition_width, partition_height = self.partition_size_x, self.partition_size_y

        while current_y < image_height:
            while current_x < image_width:
                if current_x + partition_width < image_width:
                    startx, stopx = current_x, current_x + partition_width
                else:
                    startx, stopx = image_width - partition_width, image_width
                    current_x = image_width
                if current_y + partition_height < image_height:
                    starty, stopy = current_y, current_y + partition_height
                else:
                    starty, stopy = image_width - partition_height, image_height

                partition_indices.append([[startx, stopx], [starty, stopy]])
                current_x += step[0]

            current_x = 0
            current_y += step[1]

        return partition_indices

    def test(self):
        """
        intensities = self.generate_weights(5)
        self.write_to_im(intensities, "intensities.png")
        """
        im = self.generate_weights()
        filename = "intensities" + str(self.partition_size_x) + "x" + str(self.partition_size_y) + ".png"
        self.write_to_im(im, filename)
'''
if __name__=="__main__":
    #upconv = UpConvolve([2,2],[4,4], [94,94], 5)
    #upconv.test()
    dataProv = DataProvider()
    dataProv.partition([2,2], [4,4], [94,94], 5)
'''
