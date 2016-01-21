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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse





def grab_neighbors(array, label):

    thresholded_array = _metrics.Util.threshold(array, label)
    thresholded_array_dilated = mh.dilate(thresholded_array.astype(np.uint64))

    all_neighbors = np.unique(array[thresholded_array_dilated == thresholded_array_dilated.max()].astype(np.uint64))
    all_neighbors = np.delete(all_neighbors, np.where(all_neighbors == label))

    return all_neighbors

def grab_patch(image, prob, segmentation, l, n, patch_size=(75,75)):

    borders = mh.labeled.border(segmentation, l, n)

    borders_labeled = skimage.measure.label(borders)
    
    patches = []
    
    # check if we need multiple patches because of multiple borders
    for b in range(1,borders_labeled.max()+1):
#         print 'patch for border', b
        
        border = _metrics.Util.threshold(borders_labeled, b)

        border_yx = indices = zip(*np.where(border==1))

        # fault check if no border is found
        if len(border_yx) < 2:
            print 'no border', l, n, 'border', b
            continue

        #
        # calculate border center properly
        #
        node = mh.center_of_mass(border)
        nodes = border_yx
        nodes = np.asarray(nodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        border_center = border_yx[np.argmin(dist_2)]

        # check if border_center is too close to the 4 edges
        new_border_center = [border_center[0], border_center[1]]
        if border_center[0] < patch_size[0]/2:
            return None
        if border_center[0]+patch_size[0]/2 >= segmentation.shape[0]:
            return None
        if border_center[1] < patch_size[1]/2:
            return None
        if border_center[1]+patch_size[1]/2 >= segmentation.shape[1]:
            return None

        bbox = [new_border_center[0]-patch_size[0]/2, 
                new_border_center[0]+patch_size[0]/2,
                new_border_center[1]-patch_size[1]/2, 
                new_border_center[1]+patch_size[1]/2]

        ### workaround to not sample white border of probability map
        if bbox[0] <= 33:
            return None
        if bbox[1] >= segmentation.shape[0]-33:
            return None
        if bbox[2] <= 33:
            return None
        if bbox[3] >= segmentation.shape[1]-33:
            return None


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

        output = {}
        output['id'] = str(uuid.uuid4())
        output['image'] = image[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
        output['prob'] = prob[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
        output['l'] = l
        output['n'] = n
        output['bbox'] = bbox
        output['border'] = border_yx
        output['border_center'] = new_border_center
        output['binary1'] = array1[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
        output['binary2'] = array2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
        output['overlap'] = overlap[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)

        patches.append(output)
        
    return patches
    
    

def loop(image, prob, segmentation):
    labels = range(1,len(_metrics.Util.get_histogram(segmentation.astype(np.uint64))))
    #labels = range(1,30)
    patches = []
    
    for l in labels:
        neighbors = grab_neighbors(segmentation, l)
        for n in neighbors:
            p = grab_patch(image, prob, segmentation, l, n)
            if not p:
                continue
                
            patches += p
            
    return patches

def fill_matrix(val_fn, m, patches):
    
    patch_grouper = {}
    for p in patches:
        # create key
        minlabel = min(p['l'], p['n'])
        maxlabel = max(p['l'], p['n'])
        key = str(minlabel)+'-'+str(maxlabel)

        if not key in patch_grouper:
            patch_grouper[key] = []

        patch_grouper[key] += [p]

    # now average the probabilities
    for k in patch_grouper.keys():

        weights = []
        predictions = []

        for p in patch_grouper[k]:

             weights.append(len(p['border']))
             predictions.append(test_patch(val_fn, p))

        if len(patch_grouper[k]) == 1:
            weights[0] = 1

        p_sum = 0
        w_sum = 0
        for i,w in enumerate(weights):
            # weighted arithmetic mean
            p_sum += w*predictions[i]
            w_sum += w

        p_sum /= w_sum

        patch_grouper[k] = p_sum

    for p in patch_grouper.keys():
        l = int(p.split('-')[0])
        n = int(p.split('-')[1])
        
        m[l,n] = patch_grouper[p]
        m[n,l] = patch_grouper[p]
        
    print 'merged', len(patches), 'patches into', len(patch_grouper.keys())
        
    return m

            
def setup_n():
    from test_cnn_vis import TestCNN
    t = TestCNN('7b76867e-c76a-416f-910a-7065e93c616a', 'patches_large2new')
    val_fn = t.run()
    
    return val_fn
            
def test_patch(val_fn, p):

    images = p['image'].reshape(-1, 1, 75, 75).astype(np.uint8)
    probs = p['prob'].reshape(-1, 1, 75, 75).astype(np.uint8)
    binary1s = p['binary1'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    binary2s = p['binary2'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    overlaps = p['overlap'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    targets = np.array([0], dtype=np.uint8)
    
    pred, err, acc = val_fn(images, probs, binary1s, binary2s, overlaps, targets)
            
    return pred[0][1]

def create_matrix(val_fn, segmentation, patches):
    
    
    no_labels = len(_metrics.Util.get_histogram(segmentation.astype(np.uint64)))
    
    m = np.zeros((no_labels,no_labels), dtype=np.float)
    m[:,:] = -1 # all uninitialised
 
#     for p in patches:

#         prediction = test_patch(val_fn, p)
#         m[p['l'], p['n']] = prediction
#         m[p['n'], p['l']] = prediction

    m = fill_matrix(val_fn, m, patches)

    return m

def merge(val_fn, image, seg, prob, m_old, sureness=1.):
    
    out = np.array(seg)
    m = np.array(m_old)
    
    patches = []
    
    # find largest value in matrix
    largest_index = np.where(m==m.max())
    l,n = (largest_index[0][0], largest_index[1][0])
    
    # merge these labels
    print 'merging', l, n
    out[out == l] = n
    # set matrix as merged for this entry
    m[l,:] = -2
    m[:,l] = -2
    
    # grab neighbors of l
    old_neighbors = grab_neighbors(seg, l)
    # get patches for l and n_l
    for n in old_neighbors:
        neighbors = grab_neighbors(out, n)
        for k in neighbors:
            
            # check if this is still a valid combination
            if m[n,k] == -2:
                # nope!
                continue
            
            p = grab_patch(image, prob, out, n, k)
            if not p:
                continue

            patches += p
        
    print 'recomputed', len(patches), 'patches'

    # now we have a bunch of new patches
    # -> update our matrix    
#     for p in patches:
#         prediction = test_patch(val_fn, p)
#         m[p['l'], p['n']] = prediction
#         m[p['n'], p['l']] = prediction

    m = fill_matrix(val_fn, m, patches)

    return m, out



def propagate_max_overlap(rhoana, gold):

    out = np.array(rhoana)
    
    rhoana_labels = _metrics.Util.get_histogram(rhoana.astype(np.uint64))
    
    for l,k in enumerate(rhoana_labels):
        if l == 0:
            # ignore 0 since rhoana does not have it
            continue
        values = gold[rhoana == l]
        largest_label = _metrics.Util.get_largest_label(values.astype(np.uint64))
    
        out[rhoana == l] = largest_label # set the largest label from gold here
    
    return out
        


def run(slice):

  DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
  GOLD_PATH = os.path.join(DATA_PATH,'gold/')
  RHOANA_PATH = os.path.join(DATA_PATH,'rhoana/')
  IMAGE_PATH = os.path.join(DATA_PATH,'input/')
  PROB_PATH = os.path.join(DATA_PATH,'prob/')
  PATCH_PATH = os.path.join(DATA_PATH,'test_rhoana/')
  OUTPUT_PATH = os.path.join(DATA_PATH,'fix_rhoana/')


  gold = _metrics.Util.read(GOLD_PATH+'*.tif')
  rhoana = _metrics.Util.read(RHOANA_PATH+'*.tif')
  images = _metrics.Util.read(IMAGE_PATH+'*.tif')
  probs = _metrics.Util.read(PROB_PATH+'*.tif')


  SLICE = slice

  print 'Running on slice', SLICE

  image = images[SLICE]
  prob = probs[SLICE]
  seg = _metrics.Util.normalize_labels(rhoana[SLICE])[0]

  # fill and normalize gold
  gold_zeros = _metrics.Util.threshold(gold[SLICE], 0)
  gold_filled = _metrics.Util.fill(gold[SLICE], gold_zeros.astype(np.bool))
  gold_filled_relabeled = skimage.measure.label(gold_filled).astype(np.uint64)
  gold_normalized = _metrics.Util.normalize_labels(gold_filled_relabeled)[0]

  # color both segmentations
  cm = _metrics.Util.load_colormap('/Volumes/DATA1/ac3x75/mojo/ids/colorMap.hdf5')
  colored_rhoana_normalized = cm[seg % len(cm)]
  colored_gold_normalized = cm[gold_normalized % len(cm)]



  print 'Find all patches'
  t0 = time.time()
  p = loop(image, prob, seg)
  print time.time()-t0, 'seconds for', len(p), 'patches'

  print 'Setting up CNN'
  val_fn = setup_n()  

  print 'Creating Matrix'
  m_new = create_matrix(val_fn, seg, p)
  out = seg
  sureness = 0.
  surenesses = []
  vi_s = []
  images = []
  before_VI = partition_comparison.variation_of_information(gold_normalized.ravel(), seg.ravel())

  max_counter = 0

  while m_new.max() >= sureness:
      max_counter += 1
      print 'Iteration', max_counter
      print 'Merging for sureness s=', m_new.max()
      surenesses.append(m_new.max())
      
      m_new, out = merge(val_fn, image, out, prob, m_new)
      
      vi = partition_comparison.variation_of_information(out.ravel(), gold_normalized.ravel())
      vi_s.append(vi)
      
      images.append(np.array(out))
      
      print 'New VI', vi
      print '-'*80
        
  before_VI = partition_comparison.variation_of_information(gold_normalized.ravel(), seg.ravel())

  index = vi_s.index(min(vi_s))
  print 'Before VI', before_VI
  print 'Lowest VI', vi_s[index]
  print 'Iterations', index
  print 'Sureness', surenesses[index]


  # create plot
  new_rhoana = propagate_max_overlap(seg, gold_normalized)
  target_vi = partition_comparison.variation_of_information(new_rhoana.ravel(), gold_normalized.ravel())

  bins = np.arange(0, len(vi_s))#np.arange(len(vi_s), 0, -1)
  # bins /= 100

  vi_s_ = vi_s #[x for (y,x) in sorted(zip(surenesses, vi_s))]
  surenesses_ = bins #surenesses#[y for (y,x) in sorted(zip(surenesses, vi_s))]

  original_vi = [before_VI]*len(vi_s)
  target_vi = [target_vi]*len(vi_s)

  fig, ax = plt.subplots()

  ax.plot(surenesses_, target_vi, 'k--', label='Target VI')
  ax.plot(surenesses_, vi_s_, 'k', label='Variation of Information')
  ax.plot(surenesses_, original_vi, 'k:', label='Rhoana VI before')
  # ax.set_yscale('log')

  # Now add the legend with some customizations.
  legend = ax.legend(loc='upper center', shadow=True)
  ax.set_ylim([0.6,1.5])

  # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
  frame = legend.get_frame()
  frame.set_facecolor('0.90')

  # Set the fontsize
  for label in legend.get_texts():
      label.set_fontsize('large')

  for label in legend.get_lines():
      label.set_linewidth(1.5)  # the legend line width

  plt.savefig(OUTPUT_PATH+os.sep+'graph_'+str(slice)+'.png')

  mh.imsave(OUTPUT_PATH+os.sep+'best_'+str(slice)+'.tif', images[index])
  mh.imsave(OUTPUT_PATH+os.sep+'last_'+str(slice)+'.tif', images[-1])

  print 'All done.'



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--slice", type=int, help='The slice.')

  args = parser.parse_args()

  run(args.slice)



