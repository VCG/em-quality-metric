from cnn import CNN

from util import Util
from patch import Patch
from fixer import Fixer
from uglify import Uglify

import os
import mahotas as mh
import numpy as np
import time
import skimage.measure

class UITools(object):

  @staticmethod
  def load_cnn():
    cnn = CNN('mine_merged_large_7', 'patches_7th', ['image', 'prob', 'merged_array', 'larger_border_overlap'])

    return cnn

  @staticmethod
  def get_top5_merge_errors(cnn, input_image, input_prob, input_rhoana, verbose=True):

    #
    # this creates the top bins for the best five merge splits but also simulates the user who picks the best
    #
    #
    t0 = time.time()
    fixed_volume = np.array(input_rhoana)

    merge_errors = []

    for i in range(10):
        if verbose:
          print 'working on slice', i
        
        DOJO_SLICE = i
        
        hist = Util.get_histogram(input_rhoana[DOJO_SLICE].astype(np.uint64))
        labels = range(len(hist))

        fixed_slice = np.array(input_rhoana[DOJO_SLICE], dtype=np.uint64)

        for l in labels:

            if l == 0 or hist[l]<3000:
                continue

            # single binary mask for label l
            before_merge_error = np.zeros(input_rhoana[DOJO_SLICE].shape)
            before_merge_error[fixed_slice == l] = 1

            results = Fixer.fix_single_merge(cnn,
                                              input_image[DOJO_SLICE],
                                              input_prob[DOJO_SLICE],
                                              before_merge_error, N=50, 
                                              erode=True, 
                                              invert=True,
                                              dilate=True,
                                              border_seeds=True,
                                              oversampling=False)

            if len(results) > 0:
                
                #
                # SORT THE PREDICTIONS (prediction, border)-tupels
                # LOOK AT TOP 5
                sorted_pred = sorted(results, key=lambda x: x[0])

                top5 = sorted_pred[:5]
                
                lowest_prediction = sorted_pred[0][0]
                

                # store the merge error
                # we need to store: z, l, results_no_border, borders, predictions
                merge_errors.append((i, l, lowest_prediction, (top5)))
                
    if verbose:
      print 'merge error correction done after',time.time()-t0, 'seconds'

    return merge_errors


  @staticmethod
  def get_split_patches(cnn, volume, volume_prob, volume_segmentation, oversampling=False, verbose=True, max=10000):

    bigM = []
    global_patches = []

    t0 = time.time()
    for slice in range(volume.shape[0]):

      image = volume[slice]
      prob = volume_prob[slice]
      segmentation = volume_segmentation[slice]
      groundtruth = volume_groundtruth[slice]

      
      patches = Patch.patchify(image, prob, segmentation, oversampling=oversampling, max=max)
      if verbose:
        print len(patches), 'generated in', time.time()-t0, 'seconds.'

      t0 = time.time()
      grouped_patches = Patch.group(patches)
      if verbose:
        print 'Grouped into', len(grouped_patches.keys()), 'patches in', time.time()-t0, 'seconds.'
      global_patches.append(patches)

      hist = Util.get_histogram(segmentation.astype(np.float))
      labels = len(hist)

      # create Matrix
      M = np.zeros((labels, labels), dtype=np.float)
      # .. and initialize with -1
      M[:,:] = -1



      for l_n in grouped_patches.keys():

        l = int(l_n.split('-')[0])
        n = int(l_n.split('-')[1])

        # test this patch group for l and n
        prediction = Patch.test_and_unify(grouped_patches[l_n], cnn)

        # fill value into matrix
        M[l,n] = prediction
        M[n,l] = prediction


      # now the matrix for this slice is filled
      bigM.append(M)


    bigM = np.array(bigM)
    
    return bigM, global_patches

  @staticmethod
  def get_split_error_image(input_image, input_rhoana, labels):

    if not isinstance(labels, list):
      labels = [labels]

    b = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    b[:,:,0] = input_image[:]
    b[:,:,1] = input_image[:]
    b[:,:,2] = input_image[:]
    b[:,:,3] = 255

    c = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    c[:,:,0] = input_image[:]
    c[:,:,1] = input_image[:]
    c[:,:,2] = input_image[:]
    c[:,:,3] = 255    

    d = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    d[:,:,0] = input_image[:]
    d[:,:,1] = input_image[:]
    d[:,:,2] = input_image[:]
    d[:,:,3] = 255    

    e = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    e[:,:,0] = input_image[:]
    e[:,:,1] = input_image[:]
    e[:,:,2] = input_image[:]
    e[:,:,3] = 255    



    thresholded_rhoana = Util.view_labels(input_rhoana, labels, crop=False, return_it=True)
    
    cropped_rhoana_dilated = mh.dilate(thresholded_rhoana.astype(np.uint64))
    for dilate in range(30):
      cropped_rhoana_dilated = mh.dilate(cropped_rhoana_dilated)

    cropped_rhoana_bbox = mh.bbox(cropped_rhoana_dilated)
    binary_border = mh.labeled.borders(thresholded_rhoana.astype(np.bool))

    b[input_rhoana == labels[0]] = (200,0,0,255)
    c[mh.labeled.borders(Util.threshold(input_rhoana, labels[0])) == 1] = (200,0,0,255)
    d[binary_border == 1] = (200,0,0,255)
    if len(labels) > 1:
      b[input_rhoana == labels[1]] = (0,200,0,255)
      c[mh.labeled.borders(Util.threshold(input_rhoana, labels[1])) == 1] = (0,200,0,255)

    cropped_image = Util.crop_by_bbox(input_image, cropped_rhoana_bbox)
    cropped_labels = Util.crop_by_bbox(b, cropped_rhoana_bbox)
    cropped_borders = Util.crop_by_bbox(c, cropped_rhoana_bbox)
    cropped_binary_border = Util.crop_by_bbox(d, cropped_rhoana_bbox)

    e[cropped_rhoana_bbox[0]:cropped_rhoana_bbox[1], cropped_rhoana_bbox[2]] = (255,255,0,255)
    e[cropped_rhoana_bbox[0]:cropped_rhoana_bbox[1], cropped_rhoana_bbox[3]] = (255,255,0,255)
    e[cropped_rhoana_bbox[0], cropped_rhoana_bbox[2]:cropped_rhoana_bbox[3]] = (255,255,0,255)
    e[cropped_rhoana_bbox[1], cropped_rhoana_bbox[2]:cropped_rhoana_bbox[3]] = (255,255,0,255)

    return cropped_image, cropped_labels, cropped_borders, cropped_binary_border, e




  @staticmethod
  def get_merge_error_image(input_image, input_rhoana, label, border):

    binary = Util.threshold(input_rhoana, label)
    binary_dilated = mh.dilate(binary.astype(np.bool))
    for dilate in range(30):
      binary_dilated = mh.dilate(binary_dilated)


    binary_bbox = mh.bbox(binary_dilated)
    binary_border = mh.labeled.borders(binary)

    b = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    b[:,:,0] = input_image[:]
    b[:,:,1] = input_image[:]
    b[:,:,2] = input_image[:]
    b[:,:,3] = 255

    c = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    c[:,:,0] = input_image[:]
    c[:,:,1] = input_image[:]
    c[:,:,2] = input_image[:]
    c[:,:,3] = 255        
    c[binary_border == 1] = (0,255,0,255)

    # border[binary==0] = 0

    b[border == 1] = (255,0,0,255)
    b[binary_border == 1] = (0,255,0,255)

    cropped_image = Util.crop_by_bbox(input_image, binary_bbox)
    cropped_binary_border = Util.crop_by_bbox(c, binary_bbox)
    cropped_combined_border = Util.crop_by_bbox(b, binary_bbox)
    cropped_border_only = Util.crop_by_bbox(border, binary_bbox)

    corrected_binary = UITools.correct_merge(input_rhoana, label, border)
    corrected_binary_original = np.array(corrected_binary)
    result = np.array(input_rhoana)
    corrected_binary += result.max()
    corrected_binary[corrected_binary_original == 0] = 0

    result[corrected_binary != 0] = 0
    result += corrected_binary.astype(np.uint64)
    cropped_result = Util.crop_by_bbox(corrected_binary, binary_bbox)

    return cropped_image, cropped_binary_border, cropped_combined_border, cropped_border_only, cropped_result, result

  @staticmethod
  def remove_border_mess(e):
    '''
    '''
    label_sizes = Util.get_histogram(e)
    # we only want to keep the two largest labels
    largest1 = np.argmax(label_sizes[1:])+1
    label_sizes[largest1] = 0
    largest2 = np.argmax(label_sizes[1:])+1
    label_sizes[largest2] = 0
    for l,s in enumerate(label_sizes):
        if l == 0 or s == 0:
            # this label has zero pixels anyways or is the background
            continue
        
        # find neighbor for l
        neighbors = Util.grab_neighbors(e, l)

        if largest1 in neighbors:
            # prefer the largest
            e[e==l] = largest1
        elif largest2 in neighbors:
            e[e==l] = largest2

    return e

  @staticmethod
  def correct_merge(input_rhoana, label, border):
    
    rhoana_copy = np.array(input_rhoana, dtype=np.uint64)

    # split the label using the border
    binary = Util.threshold(input_rhoana, label).astype(np.uint64)

    border[binary==0] = 0
    binary[border==1] = 2

    binary_relabeled = Util.relabel(binary)

    binary_no_border = np.array(binary_relabeled, dtype=np.uint64)
    binary_no_border[border==1] = 0
    

    sizes = mh.labeled.labeled_size(binary_no_border)
    too_small = np.where(sizes < 200)
    labeled_small = mh.labeled.remove_regions(binary_no_border, too_small)
    labeled_small_zeros = Util.threshold(labeled_small, 0)
    labeled_small = Util.fill(labeled_small, labeled_small_zeros.astype(np.bool))
    binary_no_border = Util.frame_image(labeled_small).astype(np.uint64)     
    binary_no_border[binary==0] = 0

    corrected_binary = binary_no_border

    # now let's remove the possible border mess
    n = 0
    while corrected_binary.max() != 2 and n < 6:
      corrected_binary = UITools.remove_border_mess(corrected_binary)
      corrected_binary = skimage.measure.label(corrected_binary)
      n += 1

    return corrected_binary


  @staticmethod
  def find_next_split_error(bigM):
    '''
    '''
    bigM_max = -1
    bigM_max_index = None
    bigM_max_z = -1
    for z,m in enumerate(bigM):
        if m.max() > bigM_max:
            bigM_max = m.max()
            bigM_max_indices = np.where(m == bigM_max)
            bigM_max_index = [bigM_max_indices[0][0], bigM_max_indices[1][0]]
            bigM_max_z = z

    return bigM_max_z, bigM_max_index, bigM_max


  @staticmethod
  def correct_split(cnn, m, input_image, input_prob, input_rhoana, label1, label2, oversampling=False):

    rhoana_copy = np.array(input_rhoana)

    new_m = np.array(m)

    rhoana_copy[rhoana_copy == label2] = label1

    # label2 does not exist anymore
    new_m[:,label2] = -2
    new_m[label2, :] = -2

    label1_neighbors = Util.grab_neighbors(rhoana_copy, label1)

    for l_neighbor in label1_neighbors:
      # recalculate new neighbors of l

      if l_neighbor == 0:
          # ignore neighbor zero
          continue

      prediction = Patch.grab_group_test_and_unify(cnn, input_image, input_prob, rhoana_copy, label1, l_neighbor, oversampling=oversampling)
      # print superL, l_neighbor
      new_m[label1,l_neighbor] = prediction
      new_m[l_neighbor,label1] = prediction



    return new_m, rhoana_copy

  def skip_split(m, label1, label2):

    new_m = np.array(m)

    new_m[label1, label2] = -3
    new_m[label2, label1] = -3

    return new_m






















