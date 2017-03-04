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
        self.stride_row = stride[0]
        self.stride_col = stride[1]

        self.kernel_size_x = kernel_size[0]
        self.kernel_size_y = kernel_size[1]

        self.pad_row = self.kernel_size_x - 1
        self.pad_col = self.kernel_size_y - 1

        self.kernel = np.ones((self.kernel_size_x, self.kernel_size_y))
        self.window_size = window_size
        self.conv_layers = conv_layers

    def generate_weights(self, convolution_levels):
        """
        Produces a 2D array for the given kernel size, stride and number of upconvolution layers
        Args:
            convolution_levels (int): The number of levels of upconvolution in the deep learning network.

        Returns:
            information_heatmap (numpy.array): A 2D numpy array whose entries denote the 'information' of each output pixel

        """
        current_arr = np.ones((self.kernel_size_y, self.kernel_size_x))

        if convolution_levels == 1:
            return current_arr

        for i in range(convolution_levels - 1):
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
        new_num_cols = prev_num_cols * self.stride_col + (self.kernel_size_x - self.stride_col)
        new_num_rows = prev_num_rows * self.stride_row + (self.kernel_size_y - self.stride_row)

        #Want to dilate this layer, use row-major format
        next_level = np.zeros((new_num_rows, new_num_cols))

        for row in range(prev_num_rows):
            for col in range(prev_num_cols):
                for i in range(self.kernel_size_x):
                    for j in range(self.kernel_size_y):
                        next_level[row * self.stride_row + i][col * self.stride_col + j] += previous_level_weights[row][col]

        return next_level

    def write_to_im(self, array, filename):
        """
        Writes a numpy array to an image file to allow easy viewing of the intensities
        Args:
            array (numpy.array): A 2D numpy array whose entries denote the 'information' of each input pixel
        """
        scipy.misc.imsave(filename, array)

    def test(self):
        intensities = self.generate_weights(5)
        self.write_to_im(intensities, "intensities.png")

if __name__=="__main__":
    upconv = UpConvolve([2,2],[4,4])
    upconv.test()
