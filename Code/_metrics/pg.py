import cv2
import cPickle as pickle
import mahotas as mh
import numpy as np
import os
import cv2
import cPickle as pickle
import numpy as np
import os
import random
import scipy.misc
import skimage.measure
import uuid

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

from util import Util

from scipy.spatial import distance


class PG(object):

  @staticmethod
  def rotate(patch, degrees):
      '''
      '''
      
      s = {}
      s['image'] = scipy.misc.imrotate(patch['image'], degrees, interp='nearest')
      s['prob'] = scipy.misc.imrotate(patch['prob'], degrees, interp='nearest')
      s['binary1'] = scipy.misc.imrotate(patch['binary1'], degrees, interp='nearest')
      s['binary2'] = scipy.misc.imrotate(patch['binary2'], degrees, interp='nearest')
      s['overlap'] = scipy.misc.imrotate(patch['overlap'], degrees, interp='nearest')

      return s

  @staticmethod
  def fliplr(patch):
      '''
      '''
      
      s = {}
      s['image'] = np.fliplr(patch['image'])#cv2.flip(self._meta['image'], how)
      s['prob'] = np.fliplr(patch['prob'])#cv2.flip(self._meta['prob'], how)
      s['binary1'] = np.fliplr(patch['binary1'])#cv2.flip(self._meta['label1'], how)
      s['binary2'] = np.fliplr(patch['binary2'])#cv2.flip(self._meta['label2'], how)
      s['overlap'] = np.fliplr(patch['overlap'])#cv2.flip(self._meta['overlap'], how)

      return s

  @staticmethod
  def flipud(patch):
      '''
      '''

      s = {}
      s['image'] = np.flipud(patch['image'])#cv2.flip(self._meta['image'], how)
      s['prob'] = np.flipud(patch['prob'])#cv2.flip(self._meta['prob'], how)
      s['binary1'] = np.flipud(patch['binary1'])#cv2.flip(self._meta['label1'], how)
      s['binary2'] = np.flipud(patch['binary2'])#cv2.flip(self._meta['label2'], how)
      s['overlap'] = np.flipud(patch['overlap'])#cv2.flip(self._meta['overlap'], how)

      return s    


  @staticmethod
  def grab_neighbors(array, label):

      thresholded_array = Util.threshold(array, label)
      thresholded_array_dilated = mh.dilate(thresholded_array.astype(np.uint64))

      all_neighbors = np.unique(array[thresholded_array_dilated == thresholded_array_dilated.max()].astype(np.uint64))
      all_neighbors = np.delete(all_neighbors, np.where(all_neighbors == label))

      return all_neighbors   


  @staticmethod
  def analyze_border(image, prob, segmentation, l, n, patch_size=(75,75), sample_rate=10):

      borders = mh.labeled.border(segmentation, l, n)

      #
      # treat interrupted borders separately
      #
      borders_labeled = skimage.measure.label(borders)

      border_bbox = mh.bbox(borders)

      patches = []

      patch_centers = []
      border_yx = indices = zip(*np.where(borders==1))

      if len(border_yx) < 2:
        # somehow border detection did not work
        return patches

      # always sample the middle point
      border_center = (border_yx[len(border_yx)/(2)][0], border_yx[len(border_yx)/(2)][1])
      patch_centers.append(border_center)


      if sample_rate > 1 or sample_rate == -1:
          if sample_rate > len(border_yx) or sample_rate==-1:
              samples = 1
          else:
              samples = len(border_yx) / sample_rate

          for i,s in enumerate(border_yx):
              
              if i % samples == 0:

                  sample_point = s
          
                  if distance.euclidean(patch_centers[-1],sample_point) < patch_size[0]:
                    # sample to close
                    # print 'sample to close', patch_centers[-1], sample_point
                    continue

                  patch_centers.append(sample_point)
          
      borders_w_center = np.array(borders.astype(np.uint8))

      for i,c in enumerate(patch_centers):
          


          borders_w_center[c[0],c[1]] = 10*(i+1)
          # print 'marking', c, borders_w_center.shape

      # if len(patch_centers) > 1:
      #   print 'PC', patch_centers
          
      for i,c in enumerate(patch_centers):

          # print 'pc',c
          
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
          # if skip_boundaries:
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
          array1 = Util.threshold(segmentation, l).astype(np.uint8)
          # threshold for label2
          array2 = Util.threshold(segmentation, n).astype(np.uint8)
          merged_array = array1 + array2


          

          # dilate for overlap
          dilated_array1 = np.array(array1)
          dilated_array2 = np.array(array2)
          for o in range(10):
              dilated_array1 = mh.dilate(dilated_array1.astype(np.uint64))
              dilated_array2 = mh.dilate(dilated_array2.astype(np.uint64))
          overlap = np.logical_and(dilated_array1, dilated_array2)
          overlap[merged_array == 0] = 0

          # overlap_labeled = skimage.measure.label(overlap)
          # overlap_value = overlap_labeled[37,37]
          # print overlap_value
          # overlap_thresholded = np.zeros(overlap.shape)
          # overlap_thresholded[overlap_labeled == overlap_value] = 1
          # overlap = overlap_thresholded

          

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
          output['borders_labeled'] = borders_labeled[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]]
          output['borders_w_center'] = borders_w_center[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]]

          patches.append(output)
          

      return patches

  @staticmethod
  def split(image, array, label):
      '''
      '''

      large_label = Util.threshold(array, label)

      label_bbox = mh.bbox(large_label)
      label = large_label[label_bbox[0]:label_bbox[1], label_bbox[2]:label_bbox[3]]
      image = image[label_bbox[0]:label_bbox[1], label_bbox[2]:label_bbox[3]]

      #
      # smooth the image
      #
      image = mh.gaussian_filter(image, 3.5)

      grad_x = np.gradient(image)[0]
      grad_y = np.gradient(image)[1]
      grad = np.add(np.abs(grad_x), np.abs(grad_y))
      #grad = np.add(np.abs(grad_x), np.abs(grad_y))
      grad -= grad.min()
      grad /= grad.max()
      grad *= 255
      grad = grad.astype(np.uint8)
      #imshow(grad)

      # we need 4 labels as output
      max_label = 0
      #while max_label!=3:

      coords = zip(*np.where(label==1))

      seed1 = random.choice(coords)
      seed2 = random.choice(coords)
      seeds = np.zeros(label.shape, dtype=np.uint64)
      seeds[seed1] = 1
      seeds[seed2] = 2
#         imshow(seeds)
      for i in range(10):
          seeds = mh.dilate(seeds)

      ws = mh.cwatershed(grad, seeds)
      ws[label==0] = 0

      ws_relabeled = skimage.measure.label(ws.astype(np.uint64))
      max_label = ws_relabeled.max()
      #print max_label

      large_label = np.zeros(large_label.shape, dtype=np.uint64)
      large_label[label_bbox[0]:label_bbox[1], label_bbox[2]:label_bbox[3]] = ws
      return large_label        


  @staticmethod
  def generate_correct(image, prob, label):
      '''
      '''

      labels = np.arange(len(Util.get_histogram(label)))
      np.random.shuffle(labels)

      for l in labels:

          neighbors = PG.grab_neighbors(label, l)
          for n in neighbors:

              patches = PG.analyze_border(image, prob, label, l, n, sample_rate=10)

              for s in patches:

                  yield s
                  yield PG.fliplr(s)
                  yield PG.flipud(s)
                  yield PG.rotate(s, 90)
                  yield PG.rotate(s, 180)
                  yield PG.rotate(s, 270)    


  @staticmethod
  def generate_split_error(image, prob, label, n=3):
      '''
      '''

      labels = np.arange(len(Util.get_histogram(label)))
      np.random.shuffle(labels)

      for l in labels:

          for i in range(n):

              ws = PG.split(image, label, l)
              patches = PG.analyze_border(image, prob, ws, int(ws.max()-1), int(ws.max()), sample_rate=10)

              for s in patches:

                  yield s
                  yield PG.fliplr(s)
                  yield PG.flipud(s)
                  yield PG.rotate(s, 90)
                  yield PG.rotate(s, 180)
                  yield PG.rotate(s, 270)                  

  @staticmethod
  def show(patch):

    fig = plt.figure(figsize=(5,5))
    plt.imshow(patch['image'], cmap='gray')
    fig = plt.figure(figsize=(5,5))
    plt.imshow(patch['prob'], cmap='gray')

    fig = plt.figure(figsize=(5,5))
    c = np.array(patch['binary1'], dtype=np.uint8)
    c[37,37] = 20
    plt.imshow(c)
    fig = plt.figure(figsize=(5,5))
    c = np.array(patch['binary2'], dtype=np.uint8)
    c[37,37] = 20
    plt.imshow(c)
    fig = plt.figure(figsize=(5,5))
    c = np.array(patch['overlap'], dtype=np.uint8)
    c[37,37] = 20
    plt.imshow(c)



