from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels

    ##TODO: DELETE DIAGNOSTICS
    print("BEFORE: solver.iter = " + str(iter))
    print("BEFORE: save format = " + str(save_format))

    if save_format:
        save_format = save_format.format(iter)

    ##TODO: DELETE DIAGNOSTICS
    print("AFTER: solver.iter = " + str(iter))
    print("AFTER: save format = " + str(save_format))

    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print('>>> {0} Iteration {1} loss {2}'.format(datetime.now(), iter, loss))
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print('>>> {0} Iteration {1} overall accuracy {2}'.format(datetime.now(), iter, acc))
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print('>>> {0} Iteration {1} mean accuracy {2}'.format(datetime.now(), iter, np.nanmean(acc)))
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('>>> {0} Iteration {1} mean IU {2}'.format(datetime.now(), iter, np.nanmean(iu)))
    freq = hist.sum(1) / hist.sum()
    print('>>> {0} Iteration {1} fwavacc {2}'.format(datetime.now(), iter, (freq[freq > 0] * iu[freq > 0]).sum()))

    return hist

def seg_tests(solver, save_format, dataset, layer='score', gt='label'):
    print('>>> {0} Begin seg tests'.format(datetime.now()))
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)
