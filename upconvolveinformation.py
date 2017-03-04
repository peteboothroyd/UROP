import numpy as np
import scipy.misc

class UpConvolve(object):
    """Class to calculate the relative information content of different pixels in an upconvolved image.
        In a deep learning architecture such as the U-Net or more recent delta-net up convolving from the lowest level
        inevitably means that padding has to be applied. Edge pixels in the final segmentation are generated from
        automatically generated padded pixels which leads to relatively worse performance towards the edge of the output.
        This is aimed at 2D input data.
    """
    def __init__(self, stride, kernel_size, window_size, conv_layers):
        """
        Args:
            stride ([int, int]):        List of strings with 2 elements; [stride in the x-direction, stride in the y-direction].
                                        Note normally both these values will be equal.
            kernel_size ([int, int]):   The kernel size for downsampling [x-size, y-size].
                                        Note normally both these values will be equal.
            window_size ([int, int]):   The window size which we are considering (eg. 510x510px).
            conv_layers (int):          The number of up/down convolution layers in the architecture.

        """
        self.stride_y = stride[0]
        self.stride_x = stride[1]

        self.kernel_size_x = kernel_size[0]
        self.kernel_size_y = kernel_size[1]

        self.window_size_x = window_size[0]
        self.window_size_y = window_size[1]

        self.kernel = np.ones((self.kernel_size_x, self.kernel_size_y))
        self.conv_layers = conv_layers

    def generate_weights(self):
        """
        Produces a 2D array for the given kernel size, stride and number of upconvolution layers
        Args:
            convolution_levels (int): The number of levels of upconvolution in the deep learning network.

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
        Recursive method which generates the next level of 'information' weights based upon the previous level
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
        current_x = self.window_size_x
        current_y = self.window_size_y
        step_x = self.kernel_size_x - self.stride_x
        step_y = self.kernel_size_y - self.stride_y

        for _ in range(self.conv_layers):
            current_x = (current_x - step_x) / self.stride_x
            current_y = (current_y - step_y) / self.stride_y

        return int(current_x), int(current_y)

    def test(self):
        """
        intensities = self.generate_weights(5)
        self.write_to_im(intensities, "intensities.png")
        """
        im = self.generate_weights()
        filename = "intensities" + str(self.window_size_x) + "x" + str(self.window_size_y) + ".png"
        self.write_to_im(im, filename)

if __name__=="__main__":
    upconv = UpConvolve([2,2],[4,4], [94,94], 5)
    upconv.test()
