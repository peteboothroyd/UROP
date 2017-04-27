import re
import skimage.io as imio

def read_info_file(path):
    """
    Relevant imformation from the data layer has been saved to an info.txt file. This parses the information to a usable format
    and stores it as fields in the output layer.

    Args:
        path (str): The path to the info file

    Returns:
        num_partitions_per_image (int): The number of partitions per overall image
        image_dim [(int), (int)]:       The dimensions of the overall image [x_size, y_size]
        im_coords [[(str)]]:            A list of strings, with the offsets of the various partitions
        stride [(int), (int)]:          The stride size of the convolution [x_size, y_size]
        kernel_size [(int), (int)]:     The kernel size of the convolution [x_size, y_size]
        n_conv_levels [int]:            The number of convolution levels

    """
    try:
        #Parsing info file for useful data.
        print("Reading info file...")
        info_file = open(path, "r")

        num_partitions_line_raw = info_file.readline()
        num_partitions_pattern = r"Number of partitions per image = (?P<num_partitions>\d+)"
        r = re.findall(num_partitions_pattern, num_partitions_line_raw)
        num_partitions_per_image = int(r[0])

        image_size_line_raw = info_file.readline()
        image_size_pattern = r"Image size = \[(?P<im_size_x>\d+), (?P<im_size_y>\d+)\]"
        r = re.findall(image_size_pattern, image_size_line_raw)
        image_dim = [int(r[0][0]), int(r[0][1])]

        stride_size_line_raw = info_file.readline()
        stride_size_pattern = r"Stride = \[(?P<stride_size_x>\d+), (?P<stride_size_y>\d+)\]"
        r = re.findall(stride_size_pattern, stride_size_line_raw)
        stride = [int(r[0][0]), int(r[0][1])]

        kernel_size_line_raw = info_file.readline()
        kernel_size_pattern = r"Kernel size = \[(?P<kernel_size_x>\d+), (?P<kernel_size_y>\d+)\]"
        r = re.findall(kernel_size_pattern, kernel_size_line_raw)
        kernel_size = [int(r[0][0]), int(r[0][1])]

        n_conv_levels_line_raw = info_file.readline()
        n_conv_levels_pattern = r"Convolution levels = (?P<n_conv_levels>\d+)"
        r = re.findall(n_conv_levels_pattern, n_conv_levels_line_raw)
        n_conv_levels = int(r[0])

        im_coords = info_file.readline().split()
        info_file.close()
        ''' ##DEBUG##
        #print("number of partitions per image = " + str(num_partitions_per_image) + str(type(num_partitions_per_image)))
        print("image size = " + str(image_dim))
        print("image coords = " + str(im_coords) + ". length = " + str(len(im_coords)))
        print("stride size = " + str(stride))
        print("kernel size = " + str(kernel_size))
        print("number convolutional levels = " + str(n_conv_levels))
        '''
        return num_partitions_per_image, image_dim, im_coords, stride, kernel_size, n_conv_levels

    except IOError:
        print("Could not open info file. Check that it has been created at the path= " + path)

def create_info_file(path, num_partitions, image_dim, stride, kernel_size, num_conv_levels, partition_indices):
    """
    Store relevant information which can be used later in the output layer.

    Args:
        path (str):                 The path to the info file
        num_partitions (int):       The number of partitions per image
        image_dim [(int), (int)]:   The size of the overall image [x-size, y-size]
    """
    try:
        info_file = open(path, "w+")
        info_file.write("Number of partitions per image = " + str(num_partitions) + "\n")
        info_file.write("Image size = " + str(image_dim) + "\n")
        info_file.write("Stride = " + str(stride) + "\n")
        info_file.write("Kernel size = " + str(kernel_size) + "\n")
        info_file.write("Convolution levels = " + str(num_conv_levels) + "\n")
        for j in range(len(partition_indices)):
            startx, stopx = partition_indices[j][0][0], partition_indices[j][0][1]
            starty, stopy = partition_indices[j][1][0], partition_indices[j][1][1]
            image_coords = "x{0}_y{1}".format(startx, starty)
            info_file.write(image_coords + " ")
        info_file.close()

        print("Created image file at path: " + path)
    except:
        print("Problem loading info file with path: " + path)

def find_partition_dim(path):
    try:
        #print("im_files[0] = " + str(self.im_files[0]))
        #TODO: Is this necessary? (Joining impath to relative path, has this already been done in line 37?)
        #impath = join(self.image_path, self.im_files[0])
        #print("impath = " + str(impath))

        im = imio.imread(path)
        return im.shape
    except:
        print("Problem loading image with path: " + path)
