from scipy.spatial import distance
from util import Util
import mahotas as mh
import numpy as np
import skimage.measure
import scipy.misc
import uuid
import matplotlib.pyplot as plt
import random

class Patch(object):

  @staticmethod
  def grab(image, prob, segmentation, l, n, sample_rate=10):



    # grab border between l and n
    border = mh.labeled.border(segmentation, l, n)
    
    # also grab binary mask for l
    binary_l = Util.threshold(segmentation, l)
    # .. and n
    binary_n = Util.threshold(segmentation, n)

    # analyze both borders
    patches_l = Patch.analyze_border(image, prob, binary_l, border, sample_rate=sample_rate)
    patches_n = Patch.analyze_border(image, prob, binary_n, border, sample_rate=sample_rate)

    return patches_l, patches_n

  @staticmethod
  def patchify(image, prob, segmentation, sample_rate=10, min_pixels=100):
    '''
    '''
    patches = []
    hist = Util.get_histogram(segmentation.astype(np.uint64))
    labels = len(hist)
    

    for l in range(labels):

      if l == 0:
        continue

      if hist[l] < min_pixels:
        continue

      neighbors = Util.grab_neighbors(segmentation, l)

      for n in neighbors:

        if n == 0:
          continue

        if hist[n] < min_pixels:
          continue

        # print 'grabbing', l, n
        p_l, p_n = Patch.grab(image, prob, segmentation, l, n, sample_rate)

        patches += p_l
        patches += p_n

    return patches




  @staticmethod
  def analyze_border(image, prob, binary_mask, border, patch_size=(75,75), sample_rate=10):

      patches = []

      patch_centers = []
      border_yx = indices = zip(*np.where(border==1))

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
          
      # borders_w_center = np.array(borders.astype(np.uint8))

      # for i,c in enumerate(patch_centers):
          


      #     borders_w_center[c[0],c[1]] = 10*(i+1)
      #     # print 'marking', c, borders_w_center.shape

      # # if len(patch_centers) > 1:
      # #   print 'PC', patch_centers
          
      for i,c in enumerate(patch_centers):

          # print 'pc',c
          
  #         for border_center in patch_centers:

          # check if border_center is too close to the 4 edges
          new_border_center = [c[0], c[1]]

          if new_border_center[0] < patch_size[0]/2:
              # print 'oob1', new_border_center
              # return None
              continue
          if new_border_center[0]+patch_size[0]/2 >= border.shape[0]:
              # print 'oob2', new_border_center
              # return None
              continue
          if new_border_center[1] < patch_size[1]/2:
              # print 'oob3', new_border_center
              # return None
              continue
          if new_border_center[1]+patch_size[1]/2 >= border.shape[1]:
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
          if bbox[1] >= border.shape[0]-33:
              # return None
              # print 'ppb'
              continue
          if bbox[2] <= 33:
              # return None
              # print 'ppb'
              continue
          if bbox[3] >= border.shape[1]-33:
              # return None
              # print 'ppb'
              continue

          
          cutout_border = mh.labeled.border(binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1], 1, 0)


          # cutout_border = np.array(border[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool))
          for d in range(1):
            cutout_border = mh.dilate(cutout_border)

          relabeled_cutout_border = skimage.measure.label(cutout_border)
          relabeled_cutout_border += 1 # avoid 0
          center_cutout = relabeled_cutout_border[37,37]
          relabeled_cutout_border[relabeled_cutout_border != center_cutout] = 0


          relabeled_cutout_binary_mask = skimage.measure.label(binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1])
          relabeled_cutout_binary_mask += 1
          center_cutout = relabeled_cutout_binary_mask[37,37]
          relabeled_cutout_binary_mask[relabeled_cutout_binary_mask != center_cutout] = 0


          # # threshold for label1
          # array1 = Util.threshold(segmentation, l).astype(np.uint8)
          # # threshold for label2
          # array2 = Util.threshold(segmentation, n).astype(np.uint8)
          # merged_array = array1 + array2


          

          # # dilate for overlap
          # dilated_array1 = np.array(array1)
          # dilated_array2 = np.array(array2)
          # for o in range(10):
          #     dilated_array1 = mh.dilate(dilated_array1.astype(np.uint64))
          #     dilated_array2 = mh.dilate(dilated_array2.astype(np.uint64))
          # overlap = np.logical_and(dilated_array1, dilated_array2)
          # overlap[merged_array == 0] = 0

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
          # output['binary1'] = binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
          output['binary1'] = relabeled_cutout_binary_mask.astype(np.bool)
          output['bbox'] = bbox
          output['border'] = border_yx
          output['border_center'] = new_border_center
          output['overlap'] = relabeled_cutout_border.astype(np.bool)
          # output['overlap'] = overlap[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)

          # output['borders_labeled'] = borders_labeled[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]]
          # output['borders_w_center'] = borders_w_center[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]]

          patches.append(output)
          

      return patches

  @staticmethod
  def split_label(image, binary):

    bbox = mh.bbox(binary)
    
    sub_image = np.array(image[bbox[0]:bbox[1], bbox[2]:bbox[3]])
    sub_binary = np.array(binary[bbox[0]:bbox[1], bbox[2]:bbox[3]])

    sub_binary_border = mh.labeled.borders(sub_binary, Bc=mh.disk(3))    
    
    sub_binary = mh.erode(sub_binary.astype(np.bool))
    for e in range(5):
      sub_binary = mh.erode(sub_binary)
    # # sub_binary = mh.erode(sub_binary)    
    

    if sub_image.shape[0] < 2 or sub_image.shape[1] < 2:
      return np.zeros(binary.shape, dtype=np.bool), np.zeros(binary.shape, dtype=np.bool)

    #
    # smooth the image
    #
    sub_image = mh.gaussian_filter(sub_image, 3.5)

    grad_x = np.gradient(sub_image)[0]
    grad_y = np.gradient(sub_image)[1]
    grad = np.add(np.abs(grad_x), np.abs(grad_y))

    grad -= grad.min()
    grad /= grad.max()
    grad *= 255
    grad = grad.astype(np.uint8)
    
    coords = zip(*np.where(sub_binary==1))

    if len(coords) < 2:
      # print 'STRAAAAANGE'
      return np.zeros(binary.shape, dtype=np.bool), np.zeros(binary.shape, dtype=np.bool)

    seed1 = random.choice(coords)
    seed2 = random.choice(coords)
    seeds = np.zeros(sub_binary.shape, dtype=np.uint64)
    seeds[seed1] = 1
    seeds[seed2] = 2

    for i in range(10):
      seeds = mh.dilate(seeds)

    ws = mh.cwatershed(grad, seeds)
    ws[sub_binary==0] = 0

#     ws_relabeled = skimage.measure.label(ws.astype(np.uint8))
#     ws_relabeled[sub_binary==0] = 0
#     max_label = ws_relabeled.max()
#     plt.figure()
#     imshow(ws)

    binary_mask = Util.threshold(ws, ws.max())
    border = mh.labeled.border(ws, ws.max(), ws.max()-1, Bc=mh.disk(2))
    border[sub_binary_border == 1] = 0 # remove any "real" border pixels
    
#     plt.figure()
#     imshow(binary_mask)

#     plt.figure()
#     imshow(border)

    # at this point, there can be multiple borders and labels
    labeled_border = skimage.measure.label(border)
    labeled_binary_mask = skimage.measure.label(binary_mask)
    # .. and we are going to select only the largest
    largest_border_label = Util.get_largest_label(labeled_border.astype(np.uint16), True)
    largest_binary_mask_label = Util.get_largest_label(labeled_binary_mask.astype(np.uint16), True)
    # .. filter out everything else
    border[labeled_border != largest_border_label] = 0
    binary_mask[labeled_binary_mask != largest_binary_mask_label] = 0

    
    large_label = np.zeros(binary.shape, dtype=np.bool)
    large_border = np.zeros(binary.shape, dtype=np.bool)
    large_label[bbox[0]:bbox[1], bbox[2]:bbox[3]] = binary_mask
    large_border[bbox[0]:bbox[1], bbox[2]:bbox[3]] = border
    
    return large_label, large_border


  @staticmethod
  def show(patches, cnn=None, pred_threshold=1.):

    if type(patches) != type(list()):
      patches = [patches]

    for i,patch in enumerate(patches):

      if cnn:
        pred = cnn.test_patch(patch)
        if pred < pred_threshold:
          continue

      else:
        pred = '?'

      fig = plt.figure(figsize=(2,2))

      text = '\n\n\nPatch '+str(i) + ': ' + str(pred)
      text += '\n'+str(patch['bbox'])

      fig.text(0,1,text)
      plt.imshow(patch['image'], cmap='gray')
      plt.figure(figsize=(2,2))
      plt.imshow(patch['prob'], cmap='gray')
      plt.figure(figsize=(2,2))
      plt.imshow(patch['binary1'], cmap='gray')
      plt.figure(figsize=(2,2))
      plt.imshow(patch['overlap'], cmap='gray')


  @staticmethod
  def test(patches, cnn):
    if type(patches) != type(list()):
      patches = [patches]

    results = []

    for i,patch in enumerate(patches):

      pred = cnn.test_patch(patch)
      results.append(pred)

    print 'N: ', len(results)
    print 'Min: ', np.min(results)
    print 'Max: ', np.max(results)
    print 'Mean: ', np.mean(results)
    print 'Std: ', np.std(results)
    print 'Var: ', np.var(results)

    return results





  @staticmethod
  def rotate(patch, degrees):
      '''
      '''
      
      s = {}
      s['image'] = scipy.misc.imrotate(patch['image'], degrees, interp='nearest')
      s['prob'] = scipy.misc.imrotate(patch['prob'], degrees, interp='nearest')
      s['binary1'] = scipy.misc.imrotate(patch['binary1'], degrees, interp='nearest')
      # s['binary2'] = scipy.misc.imrotate(patch['binary2'], degrees, interp='nearest')
      s['overlap'] = scipy.misc.imrotate(patch['overlap'], degrees, interp='nearest')
      s['bbox'] = patch['bbox']

      return s

  @staticmethod
  def fliplr(patch):
      '''
      '''
      
      s = {}
      s['image'] = np.fliplr(patch['image'])#cv2.flip(self._meta['image'], how)
      s['prob'] = np.fliplr(patch['prob'])#cv2.flip(self._meta['prob'], how)
      s['binary1'] = np.fliplr(patch['binary1'])#cv2.flip(self._meta['label1'], how)
      # s['binary2'] = np.fliplr(patch['binary2'])#cv2.flip(self._meta['label2'], how)
      s['overlap'] = np.fliplr(patch['overlap'])#cv2.flip(self._meta['overlap'], how)
      s['bbox'] = patch['bbox']

      return s

  @staticmethod
  def flipud(patch):
      '''
      '''

      s = {}
      s['image'] = np.flipud(patch['image'])#cv2.flip(self._meta['image'], how)
      s['prob'] = np.flipud(patch['prob'])#cv2.flip(self._meta['prob'], how)
      s['binary1'] = np.flipud(patch['binary1'])#cv2.flip(self._meta['label1'], how)
      # s['binary2'] = np.flipud(patch['binary2'])#cv2.flip(self._meta['label2'], how)
      s['overlap'] = np.flipud(patch['overlap'])#cv2.flip(self._meta['overlap'], how)
      s['bbox'] = patch['bbox']

      return s
