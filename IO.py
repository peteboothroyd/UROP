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
        image_dim [(int), (int)]:       The dimensions of the overall image
        output_im_files [[(str)]]:      A list of lists of strings, with the paths to the images saved by the data provider layer

    """
    try:
        #Parsing info file for useful data.
        info_file = open(path, "r")

        num_partitions_line_raw = info_file.readline()
        num_partitions_pattern = r"Number of partitions per image = (?P<num_partitions>\d+)"
        r = re.findall(num_partitions_pattern, num_partitions_line_raw)
        num_partitions_per_image = int(r[0])

        image_size_line_raw = info_file.readline()
        image_size_pattern = r"Image size = \[(?P<im_size_x>\d+), (?P<im_size_y>\d+)\]"
        r = re.findall(image_size_pattern, image_size_line_raw)
        image_dim = [int(r[0][0]), int(r[0][1])]

        output_im_files = [f.split() for f in info_file.readlines()]
        info_file.close()

        print("number of partitions per image = " + str(num_partitions_per_image) + str(type(num_partitions_per_image)))

        return num_partitions_per_image, image_dim, output_im_files

    except IOError:
        print("Could not open info file. Check that it has been created at the path= " + path)

def create_info_file(path, num_partitions, image_dim):
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
        info_file.write("Image size = " + str(image_dim))
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
