from scipy.spatial import distance
from util import Util
import mahotas as mh
import numpy as np
import skimage.measure
import scipy.misc
import uuid
import matplotlib.pyplot as plt
import random
import time

# from uglify import Uglify

class Patch(object):


  @staticmethod
  def grab_single(image, prob, segmentation, l, sample_rate=10, mode='', oversampling=False, patch_size=(75,75)):
    
    # also grab binary mask for l
    binary_l = Util.threshold(segmentation, l)
    binary_n = Util.threshold(binary_l, 0)

    border = mh.labeled.border(binary_l, 1, 0)

    # analyze both borders
    patches_l = Patch.analyze_border(image, prob, binary_l, binary_n, border, 1, 0, sample_rate=sample_rate, patch_size=patch_size, oversampling=oversampling, mode=mode)


    return patches_l


  @staticmethod
  def grab(image, prob, segmentation, l, n, sample_rate=10, mode='', oversampling=False, patch_size=(75,75)):



    # grab border between l and n
    border = mh.labeled.border(segmentation, l, n)
    
    # also grab binary mask for l
    binary_l = Util.threshold(segmentation, l)
    # .. and n
    binary_n = Util.threshold(segmentation, n)

    # analyze both borders
    patches_l = Patch.analyze_border(image, prob, binary_l, binary_n, border, l, n, sample_rate=sample_rate, patch_size=patch_size, oversampling=oversampling, mode=mode)
    patches_n = Patch.analyze_border(image, prob, binary_n, binary_l, border, n, l, sample_rate=sample_rate, patch_size=patch_size, oversampling=oversampling, mode=mode)

    return patches_l, patches_n

  @staticmethod
  def patchify(image, prob, segmentation, sample_rate=10, min_pixels=100, max=1000, oversampling=False, ignore_zero_neighbor=True):
    '''
    '''
    patches = []
    hist = Util.get_histogram(segmentation.astype(np.uint64))
    labels = len(hist)

    batch_count = 0
    

    for l in range(labels):

      if l == 0:
        continue

      if hist[l] < min_pixels:
        continue

      neighbors = Util.grab_neighbors(segmentation, l)

      for n in neighbors:

        if ignore_zero_neighbor and n == 0:
          continue

        if hist[n] < min_pixels:
          continue

        # print 'grabbing', l, n
        p_l, p_n = Patch.grab(image, prob, segmentation, l, n, sample_rate, oversampling=oversampling)

        patches += p_l
        patches += p_n

        if len(patches) >= max:

          return patches[0:max]        


    return patches


  @staticmethod
  def split_and_patchify(image, prob, segmentation, max=1000, n=1, min_pixels=100, sample_rate=10, oversampling=False, patch_size=(75,75)):
    '''
    '''
    hist = Util.get_histogram(segmentation.astype(np.uint64))
    labels = range(len(hist))
    np.random.shuffle(labels)

    patches = []

    for l in labels:

      if l == 0:
        continue

      for s in range(n):

        binary_mask = Util.threshold(segmentation, l)

        splitted_label, border = Patch.split_label(image, binary_mask)

        # check if splitted_label is large enough
        if len(splitted_label[splitted_label == 1]) < min_pixels:
          continue

        binary_full_mask = np.array(binary_mask)
        binary_full_mask[splitted_label==1] = 2
        
        patches_l, patches_n = Patch.grab(image, prob, binary_full_mask, 1, 2, sample_rate=sample_rate, oversampling=oversampling, patch_size=patch_size)

        patches += patches_l
        patches += patches_n

        if len(patches) >= max:

          return patches[0:max]

    return patches


  @staticmethod
  def patchify_maxoverlap(image, prob, segmentation, gold, sample_rate=3, min_pixels=10, oversampling=True,
                      ignore_zero_neighbor=True, patch_size=(31,31)):
    '''
    '''

    # fill segmentation using max overlap and relabel it
    fixed = Util.propagate_max_overlap(segmentation, gold)
    fixed = Util.relabel(fixed)

    # grab borders of segmentation and fixed
    segmentation_borders = mh.labeled.borders(segmentation)
    fixed_borders = mh.labeled.borders(fixed)
    bad_borders = segmentation_borders-fixed_borders


    patches = []
    error_patches = []
    hist = Util.get_histogram(segmentation.astype(np.uint64))
    labels = range(len(hist))
    np.random.shuffle(labels)


    batch_count = 0
    

    for l in labels:

      if l == 0:
        continue

      if hist[l] < min_pixels:
        continue

      neighbors = Util.grab_neighbors(segmentation, l)

      for n in neighbors:

        if ignore_zero_neighbor and n == 0:
          continue

        if hist[n] < min_pixels:
          continue


        # grab the border between l and n
        # t0 = time.time()
        l_n_border = mh.labeled.border(segmentation, l, n)
        # print t0-time.time()

        # check if the border is a valid split or a split error
        split_error = False
        



        # # print 'grabbing', l, n
        p_l, p_n = Patch.grab(image, prob, segmentation, l, n, sample_rate, oversampling=oversampling, patch_size=patch_size)

        for p in p_l:

          patch_center = p['border_center']
          if bad_borders[patch_center[0], patch_center[1]] == 1:
            error_patches.append(p)
          else:
            patches.append(p)


        for p in p_n:

          patch_center = p['border_center']
          if bad_borders[patch_center[0], patch_center[1]] == 1:
            error_patches.append(p)
          else:
            patches.append(p)            

        # patches += p_l
        # patches += p_n

        # if len(patches) >= max:

        #   return patches[0:max]    

    patches = patches[0:len(error_patches)]


    # we will always have less error patches

    # missing_error_patches = len(patches)-len(error_patches)

    # while missing_error_patches > 2000:

    #   new_error_patches = Patch.split_and_patchify(image, prob, fixed, 
    #                                                 min_pixels=1000,
    #                                                 max=missing_error_patches,
    #                                                 n=10, sample_rate=10, oversampling=True,
    #                                                 patch_size=(31,31))

    #   error_patches += new_error_patches
    #   missing_error_patches = len(patches)-len(error_patches)

    return error_patches, patches


  @staticmethod
  def grab_group_test_and_unify(cnn, image, prob, segmentation, l, n, sample_rate=10, oversampling=False):
    '''
    '''
    patches = []
    patches_l, patches_n = Patch.grab(image, prob, segmentation, l, n, oversampling=oversampling)
    patches += patches_l
    patches += patches_n

    grouped_patches = Patch.group(patches)

    if len(grouped_patches.keys()) != 1:
      # out of bound condition due to probability map
      return -1

    prediction = Patch.test_and_unify(grouped_patches.items()[0][1], cnn)

    return prediction


  @staticmethod
  def analyze_border(image,
                     prob,
                     binary_mask,
                     binary_mask2,
                     border,
                     l=1,
                     n=0,
                     mode='',
                     patch_size=(75,75),
                     sample_rate=10,
                     oversampling=False):

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

                  if not oversampling:
            
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
              # return None
              continue
          if new_border_center[0]+patch_size[0]/2 >= border.shape[0]:
              # return None
              continue
          if new_border_center[1] < patch_size[1]/2:
              # return None
              continue
          if new_border_center[1]+patch_size[1]/2 >= border.shape[1]:
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
              continue
          if bbox[1] >= border.shape[0]-33:
              # return None
              continue
          if bbox[2] <= 33:
              # return None
              continue
          if bbox[3] >= border.shape[1]-33:
              # return None
              continue

          
          relabeled_cutout_binary_mask = skimage.measure.label(binary_mask[bbox[0]:bbox[1] + 1,
                                                                           bbox[2]:bbox[3] + 1])
          relabeled_cutout_binary_mask += 1
          center_cutout = relabeled_cutout_binary_mask[patch_size[0]/2,patch_size[1]/2]
          relabeled_cutout_binary_mask[relabeled_cutout_binary_mask != center_cutout] = 0

          relabeled_cutout_binary_mask = relabeled_cutout_binary_mask.astype(np.bool)

          cutout_border = mh.labeled.border(relabeled_cutout_binary_mask, 1, 0)

          merged_array = binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1] + binary_mask2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
          merged_array_border = mh.labeled.border(merged_array, 1, 0)


          # cutout_border = np.array(border[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool))
          for d in range(1):
            cutout_border = mh.dilate(cutout_border)

          for d in range(1):
            merged_array_border = mh.dilate(merged_array_border)



          isolated_border = cutout_border - merged_array_border
          isolated_border[cutout_border==0] = 0          

          larger_isolated_border = np.array(isolated_border)  
          for d in range(5):
            larger_isolated_border = mh.dilate(larger_isolated_border)

          # relabeled_cutout_border = cutout_border

          # relabeled_cutout_border = skimage.measure.label(cutout_border)
          # relabeled_cutout_border += 1 # avoid 0
          # center_cutout = relabeled_cutout_border[patch_size[0]/2,patch_size[1]/2]
          # relabeled_cutout_border[relabeled_cutout_border != center_cutout] = 0


          # # threshold for label1
          # array1 = Util.threshold(segmentation, l).astype(np.uint8)
          # threshold for label2
          # array2 = Util.threshold(segmentation, n).astype(np.uint8)
          # merged_array = array1 + array2


          

          # # dilate for overlap
          dilated_array1 = np.array(binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1])
          dilated_array2 = np.array(binary_mask2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1])
          for o in range(patch_size[0]/2):
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

          dyn_obj = np.zeros(merged_array.shape)
          r = 10
          midpoint = [patch_size[0]/2, patch_size[1]/2]
          dyn_obj[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r] = merged_array[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r]

          dyn_bnd = np.zeros(overlap.shape)
          r = 25
          midpoint = [patch_size[0]/2, patch_size[1]/2]
          dyn_bnd[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r] = overlap[midpoint[0]-r:midpoint[0]+r, midpoint[1]-r:midpoint[1]+r]



          output = {}
          output['id'] = str(uuid.uuid4())
          output['image'] = image[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
          
          output['prob'] = prob[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
          # output['binary1'] = binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
          output['binary'] = relabeled_cutout_binary_mask.astype(np.bool)
          output['binary1'] = binary_mask[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
          output['binary2'] = binary_mask2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
          output['merged_array'] = merged_array.astype(np.bool)
          output['dyn_obj'] = dyn_obj.astype(np.bool)
          output['dyn_bnd'] = dyn_bnd.astype(np.bool)
          output['bbox'] = bbox
          output['border'] = border_yx
          output['border_center'] = new_border_center
          output['border_overlap'] = isolated_border.astype(np.bool)
          output['overlap'] = overlap.astype(np.bool)
          output['larger_border_overlap'] = larger_isolated_border.astype(np.bool)
          output['l'] = l
          output['n'] = n
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
    for e in range(15):
      sub_binary = mh.erode(sub_binary)
    # # sub_binary = mh.erode(sub_binary)    
    

    if sub_binary.shape[0] < 2 or sub_binary.shape[1] < 2:
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
      if 'bbox' in patch:
        text += '\n'+str(patch['bbox'])
      text += '\nimage'

      fig.text(0,1,text)
      plt.imshow(patch['image'], cmap='gray')
      fig = plt.figure(figsize=(2,2))
      fig.text(0,1,'prob')
      plt.imshow(patch['prob'], cmap='gray')
      fig = plt.figure(figsize=(2,2))
      fig.text(0,1,'binary')
      plt.imshow(patch['binary'], cmap='gray')
      if 'input_binary1' in patch:
        fig = plt.figure(figsize=(2,2))
        fig.text(0,1,'input_binary1')
        plt.imshow(patch['input_binary1'], cmap='gray')
      if 'input_binary2' in patch:
        fig = plt.figure(figsize=(2,2))
        fig.text(0,1,'input_binary2')
        plt.imshow(patch['input_binary2'], cmap='gray')
      fig = plt.figure(figsize=(2,2))
      fig.text(0,1,'merged_array')
      plt.imshow(patch['merged_array'], cmap='gray')
      fig = plt.figure(figsize=(2,2))
      fig.text(0,1,'dyn_obj')
      plt.imshow(patch['dyn_obj'], cmap='gray')         
      fig = plt.figure(figsize=(2,2))
      fig.text(0,1,'dyn_bnd')
      plt.imshow(patch['dyn_bnd'], cmap='gray')      
      fig = plt.figure(figsize=(2,2))
      fig.text(0,1,'border_overlap')
      plt.imshow(patch['border_overlap'], cmap='gray')
      fig = plt.figure(figsize=(2,2))
      fig.text(0,1,'larger_border_overlap')
      plt.imshow(patch['larger_border_overlap'], cmap='gray')


  @staticmethod
  def save_as_image(patch):

      figsize=(2,2)

      for s in patch.keys():
          
          if type(patch[s]) == type(np.zeros((1,1))) and patch[s].ndim == 2:
            # fig = plt.figure(figsize=figsize)
            # plt.imshow(patch[s], cmap='gray')                
            # plt.savefig('/tmp/'+s+'.png')
            if patch[s].dtype == np.bool or s =='dyn_obj':

              mh.imsave('/tmp/'+s+'.tif', (patch[s]*255).astype(np.uint8))

            else:

              mh.imsave('/tmp/'+s+'.tif', (patch[s]).astype(np.uint8))




  @staticmethod
  def load_and_show(p, start=0, end=10, cnn=None, pred_threshold=1.):

    patch_size = (75,75)

    reshaped_patches = []



    patch_reshaped = {
      'image': p['image'].reshape(-1, 1, patch_size[0], patch_size[1]),
      'prob': p['prob'].reshape(-1, 1, patch_size[0], patch_size[1]),
      'binary': p['binary'].astype(np.uint8).reshape(-1, 1, patch_size[0], patch_size[1])*255,
      'merged_array': p['merged_array'].astype(np.uint8).reshape(-1, 1, patch_size[0], patch_size[1])*255,
      'dyn_obj': p['dyn_obj'].astype(np.uint8).reshape(-1, 1, patch_size[0], patch_size[1])*255,
      'dyn_bnd': p['dyn_bnd'].astype(np.uint8).reshape(-1, 1, patch_size[0], patch_size[1])*255,
      'border_overlap': p['border_overlap'].astype(np.uint8).reshape(-1, 1, patch_size[0], patch_size[1])*255,
      'larger_border_overlap': p['larger_border_overlap'].astype(np.uint8).reshape(-1, 1, patch_size[0], patch_size[1])*255 
    }    

    for i in range(start,end):

      patch = {
        'image': patch_reshaped['image'][i].reshape(patch_size[0], patch_size[1]),
        'prob': patch_reshaped['prob'][i].reshape(patch_size[0], patch_size[1]),
        'binary': patch_reshaped['binary'][i].reshape(patch_size[0], patch_size[1]),
        'merged_array': patch_reshaped['merged_array'][i].reshape(patch_size[0], patch_size[1]),
        'dyn_obj': patch_reshaped['dyn_obj'][i].reshape(patch_size[0], patch_size[1]),
        'dyn_bnd': patch_reshaped['dyn_bnd'][i].reshape(patch_size[0], patch_size[1]),
        'border_overlap': patch_reshaped['border_overlap'][i].reshape(patch_size[0], patch_size[1]),
        'larger_border_overlap': patch_reshaped['larger_border_overlap'][i].reshape(patch_size[0], patch_size[1])
      }

      reshaped_patches.append(patch)

    # return reshaped_patches

    Patch.show(reshaped_patches, cnn, pred_threshold)




  @staticmethod
  def test(patches, cnn, stats=True):
    if type(patches) != type(list()):
      patches = [patches]

    results = []

    for i,patch in enumerate(patches):

      pred = cnn.test_patch(patch)
      results.append(pred)

    if stats:
      print 'N: ', len(results)
      print 'Min: ', np.min(results)
      print 'Max: ', np.max(results)
      print 'Mean: ', np.mean(results)
      print 'Std: ', np.std(results)
      print 'Var: ', np.var(results)

    return results



  @staticmethod
  def group(patches):
    '''
    groups patches by label pair
    '''

    patch_grouper = {}
    for p in patches:
        # create key
        minlabel = min(p['l'], p['n'])
        maxlabel = max(p['l'], p['n'])
        key = str(minlabel)+'-'+str(maxlabel)

        if not key in patch_grouper:
            patch_grouper[key] = []

        patch_grouper[key] += [p]

    return patch_grouper


  @staticmethod
  def test_and_unify(patches, cnn, method='weighted_arithmetic_mean'):
    '''
    '''

    if method != 'weighted_arithmetic_mean':

      # just the standard mean
      return np.mean(Patch.test(patches, cnn, stats=False))

    else:

      # weighted arithmetic mean based on border length

      weights = []
      predictions = []

      for p in patches:

          # calculate the border length based on the patch size
          bbox = p['bbox']
          valid_border_points = 0
          for c in p['border']:
              if c[0] >= bbox[0] and c[0] <= bbox[1]:
                  if c[1] >= bbox[2] and c[1] <= bbox[3]:
                      # valid border point
                      valid_border_points += 1

          pred = cnn.test_patch(p)
          weights.append(valid_border_points)
          predictions.append(pred)

      p_sum = 0
      w_sum = 0
      for i,w in enumerate(weights):
          # weighted arithmetic mean
          p_sum += w*predictions[i]
          w_sum += w

      p_sum /= w_sum

      return p_sum



  @staticmethod
  def rotate(patch, degrees):
      '''
      '''
      
      s = {}
      s['image'] = scipy.misc.imrotate(patch['image'], degrees, interp='nearest')
      s['prob'] = scipy.misc.imrotate(patch['prob'], degrees, interp='nearest')
      s['binary'] = scipy.misc.imrotate(patch['binary'], degrees, interp='nearest')
      s['input_binary1'] = scipy.misc.imrotate(patch['input_binary1'], degrees, interp='nearest')
      s['input_binary2'] = scipy.misc.imrotate(patch['input_binary2'], degrees, interp='nearest')
      s['merged_array'] = scipy.misc.imrotate(patch['merged_array'], degrees, interp='nearest')
      s['dyn-obj'] = scipy.misc.imrotate(patch['dyn-obj'], degrees, interp='nearest')
      s['dyn-bnd'] = scipy.misc.imrotate(patch['dyn-bnd'], degrees, interp='nearest')
      s['border-overlap'] = scipy.misc.imrotate(patch['border-overlap'], degrees, interp='nearest')
      s['bbox'] = patch['bbox']

      return s

  @staticmethod
  def fliplr(patch):
      '''
      '''
      
      s = {}
      s['image'] = np.fliplr(patch['image'])#cv2.flip(self._meta['image'], how)
      s['prob'] = np.fliplr(patch['prob'])#cv2.flip(self._meta['prob'], how)
      s['binary'] = np.fliplr(patch['binary'])#cv2.flip(self._meta['label1'], how)
      s['input_binary1'] = np.fliplr(patch['input_binary1'])
      s['input_binary2'] = np.fliplr(patch['input_binary2'])
      s['merged_array'] = np.fliplr(patch['merged_array'])
      s['dyn-obj'] = np.fliplr(patch['dyn-obj'])
      s['dyn-bnd'] = np.fliplr(patch['dyn-bnd'])
      s['border-overlap'] = np.fliplr(patch['border-overlap'])
      s['bbox'] = patch['bbox']

      return s

  @staticmethod
  def flipud(patch):
      '''
      '''

      s = {}
      s['image'] = np.flipud(patch['image'])#cv2.flip(self._meta['image'], how)
      s['prob'] = np.flipud(patch['prob'])#cv2.flip(self._meta['prob'], how)
      s['binary'] = np.flipud(patch['binary'])#cv2.flip(self._meta['label1'], how)
      s['input_binary1'] = np.flipud(patch['input_binary1'])
      s['input_binary2'] = np.flipud(patch['input_binary2'])
      s['merged_array'] = np.flipud(patch['merged_array'])
      s['dyn-obj'] = np.flipud(patch['dyn-obj'])
      s['dyn-bnd'] = np.flipud(patch['dyn-bnd'])
      s['border-overlap'] = np.flipud(patch['border-overlap'])
      s['bbox'] = patch['bbox']

      return s
