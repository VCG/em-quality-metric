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
PATCH_PATH = os.path.join(DATA_PATH,'patches_large/')


gold = _metrics.Util.read(GOLD_PATH+'*.tif')
images = _metrics.Util.read(IMAGE_PATH+'*.tif')
probs = _metrics.Util.read(PROB_PATH+'*.tif')


def generate_patches(_type, start_slice, end_slice, count, filename):

    #
    # patches
    #
    NO_PATCHES = count
    
    if _type == 'split':
        _type = _metrics.SplitError
        p_target = np.ones(NO_PATCHES)
    elif _type == 'correct':
        _type = _metrics.MergeError
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

    for s in _type.generate(images[start_slice:end_slice], probs[start_slice:end_slice], gold[start_slice:end_slice], n=3, thumb=False, rotate=False, flip=False, randomize_slice=True, randomize_label=True, max_per_slice=max_per_slice):
    #for s in _metrics.SplitError.generate(images, gold, 10, thumb=False, rotate=True):    

        #data.append(s)

    #     if s._meta['label1'].astype(np.uint8).max() == 0:
    #         print 'wrong egg'
    #         continue

        # now we have a correct patch
    #     cv2.imwrite(TRAINING_PATH+str(splits)+'_image.tif', s._meta['image'])
    #     cv2.imwrite(TRAINING_PATH+str(splits)+'_prob.tif', s._meta['prob'])
    #     cv2.imwrite(TRAINING_PATH+str(splits)+'_label1.tif', img_as_ubyte(s._meta['label1']))
    #     cv2.imwrite(TRAINING_PATH+str(splits)+'_label2.tif', img_as_ubyte(s._meta['label2']))
    #     cv2.imwrite(TRAINING_PATH+str(splits)+'_overlap.tif', img_as_ubyte(s._meta['overlap']))

        p_image[patches] = s._meta['image'].ravel()
        p_prob[patches] = s._meta['prob'].ravel()
        p_label1[patches] = s._meta['label1'].ravel()
        p_label2[patches] = s._meta['label2'].ravel()
        p_overlap[patches] = s._meta['overlap'].ravel()

        t1 = time.time()
        total = t1-t0

        patches += 1
        if patches % 1000 == 0:
            print 'Another 1000 generated after',total,'seconds'    

    #     if total > 100:
    #         break

        if patches >= NO_PATCHES:
            break


            
            
            
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
run(0, 65, 100000, 'train')
run(65, 70, 10000, 'val')
run(70, 75, 10000, 'test')
