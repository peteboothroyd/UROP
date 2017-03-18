import caffe
import numpy as np
import skimage.io as imio
import ast

class DeployOutputLayer(caffe.Layer):
    """
    Output layer for the deployed net, which will be used to explore overlapping partitions
    """

    def setup(self, bottom, top):
        # Check inputs.
        if len(bottom) != 1:
            raise Exception("Need one input (propagated).")

        # Params is a python dictionary with layer parameters.
        if not self.param_str:
            params = dict()
        else:
            params = eval(self.param_str)

        # Loss weight
        self.loss_weight = params.get('loss_weight', 1.0)
        # Threshold for computing classification error
        self.thresh = 0.5
        self.width = params['width']
        self.height = params['height']
        self.test_sample = 0

        self.opened_info_file = False
        print("Set up DeployOutputLayer: ")

    def reshape(self, bottom, top):
        # Difference is shape of inputs.
        pass

    def forward(self, bottom, top):
        prob  = self.sigmoid(bottom[0].data)

        if not self.opened_info_file:
            try:
                info_file = open("./output/deploy_output/info.txt", "r")
                self.batch_size = info_file.readline()
                self.num_partitions_per_image = ast.literal_eval(info_file.readline())
                self.partition_indices = ast.literal_eval(info_file.readline())
                info_file.close()

                print("batch size = " + str(self.batch_size))
                print("number of partitions per image = " + str(self.num_partitions_per_image) + str(type(self.num_partitions_per_image)))
                print("type of partition indices = " + str(type(self.partition_indices)))

                self.opened_info_file = True
            except IOError:
                print("Could not open info file. Check that it has been created at the path: ./output/deploy_output/info.txt")

        try:
            to_save = np.zeros((self.height, self.width,3), dtype=np.float32)
        except:
            print(self.width + self.height)

        to_save[:, :, 0] = prob
        to_save[:, :, 1] = prob
        to_save[:, :, 2] = prob

        current_partition_number = int(self.test_sample / self.num_partitions_per_image)
        startx, starty = self.partition_indices[current_partition_number][0][0], self.partition_indices[current_partition_number][1][0]
        print("Current partition number = " + str(current_partition_number))
        print("Startx: " + str(startx) + ". Starty: " + str(starty))

        imio.imsave("./output/deploy_output/{0}_output{1}_y{2}.png".format(self.test_sample, startx, starty), to_save)
        self.test_sample += 1
        print("Sample number: " + str(self.test_sample))

    def sigmoid(self, x):
        """Numerically-stable sigmoid function."""
        ret = np.zeros_like(x)
        idx = x >= 0
        z = np.exp(-x[idx])
        ret[idx] = 1 / (1 + z)
        idx = x < 0
        z = np.exp(x[idx])
        ret[idx] = z / (1 + z)
        return ret
