import re

def read():
    info_file = open("info.txt", "r")

    num_partitions_line_raw = info_file.readline()
    num_partitions_pattern = r"Number of partitions per image = (?P<num_partitions>\d+)"
    r = re.findall(num_partitions_pattern, num_partitions_line_raw)
    num_partitions_per_image = int(r[0])
    print("Parsed number of images as: " + str(num_partitions_per_image))

    image_size_line_raw = info_file.readline()
    image_size_pattern = r"Image size = \[(?P<im_size_x>\d+), (?P<im_size_y>\d+)\]"
    r = re.findall(image_size_pattern, image_size_line_raw)
    im_size_x, im_size_y = int(r[0][0]), int(r[0][1])
    print("Image size x: " + str(im_size_x) + ". Image size y: " + str(im_size_y))

def write():
    file = open("info.txt", "w+")
    file.write("Number of partitions per image = " + str(10000) + "\n")
    file.write("Image size = " + str([94, 96]))
    file.close()

if __name__ == "__main__":
    write()
    read()
