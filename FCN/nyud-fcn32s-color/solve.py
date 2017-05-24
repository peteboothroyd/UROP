import caffe
import surgery, score

import numpy as np
import os
import sys

def main():
    try:
        import setproctitle
        setproctitle.setproctitle(os.path.basename(os.getcwd()))
    except:
        pass
    
    # init
    caffe.set_device(int(sys.argv[1]))
    caffe.set_mode_gpu()
    
    solver = caffe.SGDSolver('solver.prototxt')
    #solver.net.copy_from(weights)
    
    print("Transplanting weights:")
    weights = '../ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'
    proto = "../ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt"
    base_net = caffe.Net(proto, weights, caffe.TEST)
    surgery.transplant(solver.net, base_net, suffix='color')
    del base_net
    print("Transplanted weights!")
    
    # surgeries
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)
    
    # scoring
    with open('/media/ssd500/autocity_dataset/image_train.txt', 'r') as fid:
        im_files = [f.strip() for f in fid.readlines()]
    
    for _ in range(50):
        solver.step(2000)
        #score.seg_tests(solver, True, im_files, layer='score')

if __name__ == "__main__":
    main()
