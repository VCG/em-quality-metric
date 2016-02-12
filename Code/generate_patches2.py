import _metrics


import cv2
import glob
import numpy as np
import mahotas as mh
import os
import uuid
import tifffile as tif
from scipy import ndimage as nd
from scipy.misc import imrotate
import skimage.measure
from skimage import img_as_ubyte
import random
import cPickle as pickle
import time


DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
GOLD_PATH = os.path.join(DATA_PATH,'gold/')
IMAGE_PATH = os.path.join(DATA_PATH,'input/')
PROB_PATH = os.path.join(DATA_PATH,'prob/')
PATCH_PATH = os.path.join(DATA_PATH,'patches_large_sr2_small/')


gold = _metrics.Util.read(GOLD_PATH+'*.tif')
images = _metrics.Util.read(IMAGE_PATH+'*.tif')
probs = _metrics.Util.read(PROB_PATH+'*.tif')


def generate_patches(_type, start_slice, end_slice, count, filename):

    #
    # patches
    #
    NO_PATCHES = count
    
    if _type == 'split':
        _type = _metrics.PG.generate_split_error
        p_target = np.ones(NO_PATCHES)
    elif _type == 'correct':
        _type = _metrics.PG.generate_correct
        p_target = np.zeros(NO_PATCHES)        

    t0 = time.time()
    patches = 0
    data = []

    PATCH_BYTES = 75*75
    p_image = np.zeros((NO_PATCHES, PATCH_BYTES),dtype=np.uint8)
    p_prob = np.zeros((NO_PATCHES, PATCH_BYTES),dtype=np.uint8)
    p_label1 = np.zeros((NO_PATCHES, PATCH_BYTES),dtype=np.uint8)
    p_label2 = np.zeros((NO_PATCHES, PATCH_BYTES),dtype=np.uint8)
    p_overlap = np.zeros((NO_PATCHES, PATCH_BYTES),dtype=np.uint8)

    max_per_slice = int(count / (end_slice - start_slice)) + 1
    print 'Max per slice', max_per_slice

    for z in range(start_slice, end_slice):

        print 'working on slice', z

        # fill and normalize gold
        gold_zeros = _metrics.Util.threshold(gold[z], 0)
        gold_filled = _metrics.Util.fill(gold[z], gold_zeros.astype(np.bool))
        gold_filled_relabeled = skimage.measure.label(gold_filled).astype(np.uint64)
        gold_normalized = _metrics.Util.normalize_labels(gold_filled_relabeled)[0].astype(np.uint64)


        slice_counter = 0

        for s in _type(images[z], probs[z], gold_normalized):

            slice_counter += 1

            patches += 1

            if patches >= NO_PATCHES:
                break            

            if slice_counter >= max_per_slice:
                break

            p_image[patches] = s['image'].ravel()
            p_prob[patches] = s['prob'].ravel()
            p_label1[patches] = s['binary1'].ravel()
            p_label2[patches] = s['binary2'].ravel()
            p_overlap[patches] = s['overlap'].ravel()



            t1 = time.time()
            total = t1-t0


            if patches % 1000 == 0:
                print 'Another 1000 generated after',total,'seconds'    

        if patches >= NO_PATCHES:
            break         

    #     if total > 100:
    #         break




            
            
            
    print total, 'seconds for', patches, _type, 'patches'
    
    return p_image, p_prob, p_label1, p_label2, p_overlap, p_target

def shuffle_in_unison_inplace(a, b, c, d, e, f):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p], e[p], f[p]

def run(start_slice, end_slice, count, filename):
    
    correct = generate_patches('correct', start_slice, end_slice, count, filename)
    split = generate_patches('split', start_slice, end_slice, count, filename)
    
    # concatenate arrays
    combined_images = np.concatenate((correct[0], split[0]))
    combined_probs = np.concatenate((correct[1], split[1]))
    combined_label1s = np.concatenate((correct[2], split[2]))
    combined_label2s = np.concatenate((correct[3], split[3]))
    combined_overlaps = np.concatenate((correct[4], split[4]))
    combined_targets = np.concatenate((correct[5], split[5]))
    
    shuffled = shuffle_in_unison_inplace(combined_images, combined_probs, combined_label1s, combined_label2s, combined_overlaps, combined_targets)
    
    print 'saving..'
    np.savez(PATCH_PATH+filename+'.npz', image=shuffled[0], prob=shuffled[1], binary1=shuffled[2], binary2=shuffled[3], overlap=shuffled[4])
    np.savez(PATCH_PATH+filename+'_targets.npz', targets=shuffled[5])
    print 'Done!'
    
    return correct, split, shuffled

#
#
#
#
#
run(0, 65, 1000, 'train')
run(65, 70, 50, 'val')
run(70, 75, 50, 'test')
