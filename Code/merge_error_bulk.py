
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


import networkx as nx
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
import partition_comparison
import sys
import cPickle as pickle

DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
GOLD_PATH = os.path.join(DATA_PATH,'gold/')
RHOANA_PATH = os.path.join(DATA_PATH,'rhoana/')
IMAGE_PATH = os.path.join(DATA_PATH,'input/')
PROB_PATH = os.path.join(DATA_PATH,'prob/')
PATCH_PATH = os.path.join(DATA_PATH,'test_rhoana/')


# gold = _metrics.Util.read(GOLD_PATH+'*.tif')
# rhoana = _metrics.Util.read(RHOANA_PATH+'*.tif')
# images = _metrics.Util.read(IMAGE_PATH+'*.tif')
# probs = _metrics.Util.read(PROB_PATH+'*.tif')






SLICE=70
image, prob, gold, rhoana = _metrics.Util.read_section(SLICE)


# image = images[SLICE]
# prob = probs[SLICE]
seg = _metrics.Util.normalize_labels(rhoana)[0]#[SLICE])[0]

# fill and normalize gold
gold_zeros = _metrics.Util.threshold(gold,0)#[SLICE], 0)
gold_filled = _metrics.Util.fill(gold, gold_zeros.astype(np.bool))
gold_filled_relabeled = skimage.measure.label(gold_filled).astype(np.uint64)
gold_normalized = _metrics.Util.normalize_labels(gold_filled_relabeled)[0]

# color both segmentations
cm = _metrics.Util.load_colormap('/Volumes/DATA1/ac3x75/mojo/ids/colorMap.hdf5')
colored_rhoana_normalized = cm[seg % len(cm)]
colored_gold_normalized = cm[gold_normalized % len(cm)]


def get_real_border(image, prob, segmentation, l, n, patch_size=(75,75), skip_boundaries=True):

    # image = i_patch['image']
    # prob = i_patch['prob']
    # segmentation = i_patch['seg']

    borders = mh.labeled.border(segmentation, l, n)
    border_yx = indices = zip(*np.where(borders==1))

    patch_centers = []

    if len(border_yx) < 2:

      return None

    border_center = (border_yx[len(border_yx)/(2)][0], border_yx[len(border_yx)/(2)][1])
    patch_centers.append(border_center)

        
    for i,c in enumerate(patch_centers):

        
#         for border_center in patch_centers:

        # check if border_center is too close to the 4 edges
        new_border_center = [c[0], c[1]]

        if new_border_center[0] < patch_size[0]/2:
            # print 'oob1', new_border_center
            # return None
            continue
        if new_border_center[0]+patch_size[0]/2 >= segmentation.shape[0]:
            # print 'oob2', new_border_center
            # return None
            continue
        if new_border_center[1] < patch_size[1]/2:
            # print 'oob3', new_border_center
            # return None
            continue
        if new_border_center[1]+patch_size[1]/2 >= segmentation.shape[1]:
            # print 'oob4', new_border_center
            # return None
            continue
        # print new_border_center, patch_size[0]/2, border_center[0] < patch_size[0]/2

        # continue


        bbox = [new_border_center[0]-patch_size[0]/2, 
                new_border_center[0]+patch_size[0]/2,
                new_border_center[1]-patch_size[1]/2, 
                new_border_center[1]+patch_size[1]/2]

        ### workaround to not sample white border of probability map
        if skip_boundaries:
            if bbox[0] <= 33:
                # return None
                # print 'ppb'
                continue
            if bbox[1] >= segmentation.shape[0]-33:
                # return None
                # print 'ppb'
                continue
            if bbox[2] <= 33:
                # return None
                # print 'ppb'
                continue
            if bbox[3] >= segmentation.shape[1]-33:
                # return None
                # print 'ppb'
                continue

        

        # threshold for label1
        array1 = _metrics.Util.threshold(segmentation, l).astype(np.uint8)
        # threshold for label2
        array2 = _metrics.Util.threshold(segmentation, n).astype(np.uint8)
        merged_array = array1 + array2


        

        # dilate for overlap
        dilated_array1 = np.array(array1)
        dilated_array2 = np.array(array2)
        for o in range(10):
            dilated_array1 = mh.dilate(dilated_array1.astype(np.uint64))
            dilated_array2 = mh.dilate(dilated_array2.astype(np.uint64))
        overlap = np.logical_and(dilated_array1, dilated_array2)
        overlap[merged_array == 0] = 0




        patch = {}
        patch['image'] = image[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
        patch['prob'] = prob[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
        patch['binary1'] = array1[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
        patch['binary2'] = array2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
        patch['overlap'] = overlap[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
        patch['border'] = border_yx
        patch['bbox'] = bbox
        patch['border_center'] = new_border_center

        return patch


def create_merge_splits(i_patch):

    image = i_patch['image']
    prob = i_patch['prob']
    seg = i_patch['seg']
    
    patches = []
    
    #g_image = mh.gaussian_filter(image, 3.5)
    g_image = image

    grad_x = np.gradient(g_image)[0]
    grad_y = np.gradient(g_image)[1]
    grad = np.sqrt(np.add(grad_x*grad_x, grad_y*grad_y))
    grad -= grad.min()
    grad /= (grad.max() - grad.min())
    grad *= 255
    grad = grad.astype(np.uint8)

    G=nx.Graph()
    for y in range(grad.shape[0]):
        for x in range(grad.shape[1]):

            vertex_name = str(y)+'-'+str(x)
            left_vertex_name = str(y)+'-'+str(x-1)
            top_vertex_name = str(y-1)+'-'+str(x)
            G.add_node(vertex_name)

            if x>0:
                G.add_edge(left_vertex_name, vertex_name, weight=int(grad[y,x]))

            if y>0:
                G.add_edge(top_vertex_name, vertex_name, weight=int(grad[y,x]))

    starts = range(0,75,10)
    starts.append(74)
    ends = list(reversed(range(0,75,10)))
    ends = [74] + ends

    
    out = np.array(seg)

    for i in range(len(starts)):

        start = starts[i]
        end = ends[i]

        for sw in range(0,2):


            if sw == 0:
                start_v = str(start)+'-0'
                end_v = str(end)+'-74'
            else:
                start_v = '0-'+str(start)
                end_v = '74-'+str(end)

            if i == 0 and sw>0:
                # calculate first border only once
                continue

            if i == len(starts)-1 and sw>0:
                # and last one only once
                continue


            out = np.zeros(grad.shape)

#             fig = plt.figure()
            sp = nx.dijkstra_path(G, start_v, '37-37')
            sp2 = nx.dijkstra_path(G, '37-37', end_v)

            border = []
        
            for s in sp:

                y,x = s.split('-')
                out[int(y), int(x)] = 1
                border.append((int(y), int(x)))
            for s in sp2:

                y,x = s.split('-')
                out[int(y), int(x)] = 1    
                border.append((int(y), int(x)))
                
#             plt.imshow(out)
            
            #
            # create label images for our patch
            #
            patch_new_labeled = skimage.measure.label(out)
            patch_new_labeled[out == 1] = 1
            patch_new_labeled[seg == 0] = 0
            patch_new_labeled = mh.labeled.relabel(patch_new_labeled)[0]
            array1 = _metrics.Util.threshold(patch_new_labeled, 1)
            array2 = _metrics.Util.threshold(patch_new_labeled, 2)
            merged_array = array1 + array2

            dilated_array1 = np.array(array1)
            dilated_array2 = np.array(array2)
            for o in range(10):
                dilated_array1 = mh.dilate(dilated_array1.astype(np.uint64))
                dilated_array2 = mh.dilate(dilated_array2.astype(np.uint64))
            overlap = np.logical_and(dilated_array1, dilated_array2)
            overlap[merged_array == 0] = 0
            
            patch = {}
            patch['image'] = image
            patch['prob'] = prob
            patch['binary1'] = array1.astype(np.bool)
            patch['binary2'] = array2.astype(np.bool)
            patch['overlap'] = overlap.astype(np.bool)
            patch['border'] = border
            patch['bbox'] = i_patch['bbox']
            patch['border_center'] = i_patch['border_center']
            patches.append(patch)
            
    return patches

    
def create_patches_from_label_id(image, prob, seg, label_id):

    patches = []
    
    #label_bbox = mh.bbox(_metrics.Util.threshold(seg, label_id))

    isolated_label = _metrics.Util.threshold(seg, label_id)#[label_bbox[0]:label_bbox[1], label_bbox[2]:label_bbox[3]]
    
    eroded_isolated_label = np.array(isolated_label)
    for i in range(10):
        eroded_isolated_label = mh.erode(eroded_isolated_label.astype(np.bool))

    potential_center_points = zip(*np.where(eroded_isolated_label == 1))
    for p in potential_center_points:                         

        # check if patch is possible
        if p[0] <= 37 or p[0] >= eroded_isolated_label.shape[0]-37:
            # not possible
            continue
        if p[1] <= 37 or p[1] >= eroded_isolated_label.shape[1]-37:
            # also not possible
            continue

        ### workaround to not sample white border of probability map


        bbox = [p[0]-37, p[0]+37+1, p[1]-37, p[1]+37+1]


        if bbox[0] <= 33:
            # return None
            # print 'ppb'
            continue
        if bbox[1] >= seg.shape[0]-33:
            # return None
            # print 'ppb'
            continue
        if bbox[2] <= 33:
            # return None
            # print 'ppb'
            continue
        if bbox[3] >= seg.shape[1]-33:
            # return None
            # print 'ppb'
            continue


        
        patch = {}
        patch['image'] = image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        patch['prob'] = prob[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        patch['seg'] = isolated_label[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(np.bool)
        patch['bbox'] = bbox
        patch['border_center'] = (p[0], p[1])
        
        patches.append(patch)
        
    return patches    
    

def setup_n():
    from test_cnn_vis import TestCNN
    t = TestCNN('7b76867e-c76a-416f-910a-7065e93c616a', 'patches_large2new')
    val_fn = t.run()
    
    return val_fn

def test_patch(val_fn, p):
    # print p['prob'].shape
    # print p['overlap'].shape
    images = p['image'].reshape(-1, 1, 75, 75).astype(np.uint8)
    probs = p['prob'].reshape(-1, 1, 75, 75).astype(np.uint8)
    binary1s = p['binary1'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    binary2s = p['binary2'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    overlaps = p['overlap'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    targets = np.array([0], dtype=np.uint8)
    
    pred, err, acc = val_fn(images, probs, binary1s, binary2s, overlaps, targets)
            
    return pred[0][1]


def grab_neighbors(array, label):

    thresholded_array = _metrics.Util.threshold(array, label)
    thresholded_array_dilated = mh.dilate(thresholded_array.astype(np.uint64))

    all_neighbors = np.unique(array[thresholded_array_dilated == thresholded_array_dilated.max()].astype(np.uint64))
    all_neighbors = np.delete(all_neighbors, np.where(all_neighbors == label))

    return all_neighbors      



def create_merge_error(seg, l):
    
    #copy seg
    new_seg = np.array(seg)
    
    # grab one neighbor
    neighbors = grab_neighbors(new_seg, l)
    #neighbor = np.random.choice(neighbors)
    
    found_good = False
    good_n = -1

    for n in neighbors:
      if len(new_seg[new_seg == n]) > 3000:
        continue
      else:
        found_good = True
        good_n = n
        break

    if not found_good:
      return None, None, None, None, None
    else:
      neighbor = good_n

    labelA = _metrics.Util.threshold(new_seg, l)
    labelB = _metrics.Util.threshold(new_seg, neighbor)
    
    merged = labelA + labelB
    
    new_seg[merged == 1] = l
    
    return labelA, labelB, merged, new_seg, neighbor


#
#
#
#
val_fn = setup_n()


hist = _metrics.Util.get_histogram(gold_normalized.astype(np.uint64))

small_labels = np.where(hist < 3000)[0]

# no_labels = len(_metrics.Util.get_histogram(gold_normalized.astype(np.uint64)))
# labels = np.arange(no_labels)
# shuffle them
# np.random.shuffle(labels)

print len(small_labels)


results = []

for count,l in enumerate(small_labels[0:30]):



  labelA, labelB, merged, new_seg, n = create_merge_error(gold_normalized, l)

  if labelA == None:
    # didnt find a small pair of label neighbors
    continue

  print 'Introduced error between', l, n

  BBOX = mh.bbox(_metrics.Util.threshold(new_seg, l))

  patches = create_patches_from_label_id(image, prob, new_seg, l)

  print '-'*80
  print 'Working on', l
  print 'Generated', len(patches), 'patches'
  print 'Testing them now...'

  smallest_prediction = np.inf
  detected = []

  for i,p in enumerate(patches):
      split_patches = create_merge_splits(p)


      for j,p in enumerate(split_patches):
      
          # print j,p,len(split_patches)

          prediction = test_patch(val_fn, p)
          smallest_prediction = min(smallest_prediction, prediction)
          if prediction <= 0.5:
              # print 'HERE', i, prediction
              detected.append((p, prediction))
              
      if i % 300 == 0:
          print '  Another 300 input patches done..'


  # add also the real boundary
  p_real = get_real_border(image, prob, gold_normalized, l, n)
  if p_real:
    print 'adding real border for', l, n
    prediction = test_patch(val_fn, p_real)
    print 'tested as', prediction
    detected.append((p_real, prediction))



  #
  # paint the borders
  #
  painted_borders = np.array(merged)


  detected_sorted = sorted(detected, key=lambda x: x[1])

  for p in detected_sorted:
      prediction = p[1]
      p = p[0]
  #     if prediction > 0.001:
  #         continue
      
      border = p['border']
      bbox = p['bbox']
      center = p['border_center']
      for c in border:
          c_ = (center[0]+c[0]-37, center[1]+c[1]-37)
          painted_borders[c_[0], c_[1]] = 0


          
  painted_borders[merged == 0] = 0



  # store the stuff
  result_dict = {}
  result_dict['BBOX'] = BBOX
  result_dict['labelA'] = labelA[BBOX[0]:BBOX[1], BBOX[2]:BBOX[3]]
  result_dict['labelB'] = labelB[BBOX[0]:BBOX[1], BBOX[2]:BBOX[3]]
  result_dict['merged'] = merged[BBOX[0]:BBOX[1], BBOX[2]:BBOX[3]]
  result_dict['painted'] = painted_borders[BBOX[0]:BBOX[1], BBOX[2]:BBOX[3]]
  result_dict['detected_sorted'] = detected_sorted
  result_dict['real_border'] = p_real

  results.append(result_dict)


# if count % 10 == 0:
  # store every 10

pickle.dump(results, open("/Volumes/DATA1/EMQM_DATA/ac3x75/merge_errors/results"+str(count)+".p", 'wb'))

# results = []

print 'Stored and flushed.'

  # pickle.dump(results, open("/Volumes/DATA1/EMQM_DATA/ac3x75/merge_errors/results"+str(count+1)+".p", 'wb'))
  # print 'All done!'
