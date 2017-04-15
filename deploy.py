import sys
import upconvolveinformation

def main(image_path, partition_size, batch_size):
    """
        Creates a deployment file with the correct number of iterations for the caffe declaration.
        Args:
            image_path (string): A string with the path to the folder containing the images
            partition_size ((int, int)): A tuple containing the dimensions of the partition size which the image will
            be split into
            batch_size (int): The number of images to be processed
        """
        pass
        deploy_prototxt = "xxx{0}"


if __name__ == "__main__":
    #TODO: Check if this is correct
    if len(sys.argv) != 5:
        print("Must provide 4 arguments: a path to the folder containing the images (str), " +
              ", the dimensions of the partition provided (2x int) and the batch size")
    else:
        image_path, partition_size, batch_size = sys.argv[1], (sys.argv[2], sys.argv[3]), sys.argv[4]
        main(image_path, partition_size, batch_size)


