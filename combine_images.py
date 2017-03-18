class combine(object):
    def __init__(self):
        info_file = open("./output/deploy_output/info.txt")
        self.batch_size = info_file.readline()
        self.num_partitions_per_image = info_file.readline()
        partition_indices_string = info_file.readline()
        self.partition_indices = partition_indices_string.split()
        info_file.close()

    def combine_outputs(self):
        pass
