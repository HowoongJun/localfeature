#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
import importlib
import time
import cv2
from tqdm import tqdm
import pickle

from model.lfnet.mydatasets import *

from model.lfnet.det_tools import *
from model.lfnet.eval_tools import draw_keypoints
from model.lfnet.common.tf_train_utils import get_optimizer

from model.lfnet.inference import *
from model.lfnet.utils import embed_breakpoint, print_opt
from model.lfnet.common.argparse_utils import *


MODEL_PATH = './localfeature_ref/lfnet/models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)


def build_networks(config, photo, is_training):

    DET = importlib.import_module(config.detector)
    detector = DET.Model(config, is_training)

    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photo)

    heatmaps, det_endpoints = build_detector_helper(config, detector, photo)

    # extract patches
    kpts = det_endpoints['kpts']
    batch_inds = det_endpoints['batch_inds']

    kp_patches = build_patch_extraction(config, det_endpoints, photo)

    # Descriptor
    DESC = importlib.import_module(config.descriptor)
    descriptor = DESC.Model(config, is_training)
    desc_feats, desc_endpoints = descriptor.build_model(kp_patches, reuse=False) # [B*K,D]

    # scale and orientation (extra)
    scale_maps = det_endpoints['scale_maps']
    ori_maps = det_endpoints['ori_maps'] # cos/sin
    degree_maps, _ = get_degree_maps(ori_maps) # degree (rgb psuedo color code)
    kpts_scale = det_endpoints['kpts_scale']
    kpts_ori = det_endpoints['kpts_ori']
    kpts_ori = tf.atan2(kpts_ori[:,1], kpts_ori[:,0]) # radian

    ops = {
        'photo': photo,
        'is_training': is_training,
        'kpts': kpts,
        'feats': desc_feats,
        # EXTRA
        'scale_maps': scale_maps,
        'kpts_scale': kpts_scale,
        'degree_maps': degree_maps,
        'kpts_ori': kpts_ori,
    }

    return ops

def build_detector_helper(config, detector, photo):


    # if config.detector == 'resnet_detector':
    #     heatmaps, det_endpoints = build_deep_detector(config, detector, photo, reuse=False)
    # elif config.detector == 'mso_resnet_detector':
    if config.use_nms3d:
        heatmaps, det_endpoints = build_multi_scale_deep_detector_3DNMS(config, detector, photo, reuse=False)
    else:
        heatmaps, det_endpoints = build_multi_scale_deep_detector(config, detector, photo, reuse=False)
    # else:
    #     raise ValueError()
    return heatmaps, det_endpoints

def main(config, photo):

    # Build Networks
    tf.reset_default_graph()

    photo_ph = tf.placeholder(tf.float32, [1, None, None, 1]) # input grayscale image, normalized by 0~1
    is_training = tf.constant(False) # Always False in testing

    ops = build_networks(config, photo_ph, is_training)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True 
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    # load model
    saver = tf.train.Saver()

    if os.path.isdir(config.model):
        checkpoint = tf.train.latest_checkpoint(config.model)
        model_dir = config.model
    else:
        checkpoint = config.model
        model_dir = os.path.dirname(config.model)

    if checkpoint is not None:
        print('Checkpoint', os.path.basename(checkpoint))
        print("[{}] Resuming...".format(time.asctime()))
        saver.restore(sess, checkpoint)
    else:
        raise ValueError('Cannot load model from {}'.format(model_dir))    

    avg_elapsed_time = 0
    height, width = photo.shape[1:]
    photo = photo / 255.0 # normalize 0-1
    photo = np.expand_dims(photo, axis=3)
    assert photo.ndim == 4 # [1,H,W,1]

    feed_dict = {
        photo_ph: photo,
    }

    fetch_dict = {
        'kpts': ops['kpts'],
        'feats': ops['feats'],
    }
    outs = sess.run(fetch_dict, feed_dict=feed_dict)
    return outs['kpts'], outs['feats'], None

def run(image, threshold = 1000):

    parser = get_parser()

    tmp_config, unparsed = get_config(parser)

    model = './localfeature_ref/lfnet/release/models/outdoor/'
    tmp_config.model = model
    tmp_config.top_k = threshold
    if os.path.isdir(model):
        config_path = os.path.join(model, 'config.pkl')
    else:
        config_path = os.path.join(os.path.dirname(model), 'config.pkl')
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
            print_opt(config)
    except:
        raise ValueError('Fail to open {}'.format(config_path))

    for attr, dst_val in sorted(vars(tmp_config).items()):
        if hasattr(config, attr):
            src_val = getattr(config, attr)
            if src_val != dst_val:
                setattr(config, attr, dst_val)
        else:
            setattr(config, attr, dst_val)

    return main(config, image)
