import time
import mahotas as mh
import numpy as np
import skimage.measure
import random

from patch import Patch
from util import Util

class Fixer(object):

  @staticmethod
  def splits_global_from_M_automatic(cnn, bigM, volume, volume_prob, volume_segmentation, volume_groundtruth=np.zeros((1,1)), sureness_threshold=0.95, smallest_first=False, oversampling=False, verbose=True, max=10000):
    '''
    '''

    bigM = np.array(bigM)
    # for development, we just need the matrix and the patches
    # return bigM, None, global_patches

    out_volume = np.array(volume_segmentation)
    # return out_volume

    good_fix_counter = 0
    bad_fix_counter = 0
    # error_rate = 0
    fixes = []
    vi_s_30mins = []

    superMax = -np.inf
    j = 0 # minute counter
    # for i in range(60): # no. corrections in 1 minute
    #for i in range(17280): # no. corrections in 24 h
    i = 0
    time_counter = 0
    while True: # no time limit
      # print 'Correction', i

      if (j>0 and j % 30 == 0):
        # compute VI every 30 minutes
        vi_after_30_min = []
        for ov in range(out_volume.shape[0]):
            vi = Util.vi(volume_groundtruth[ov], out_volume[ov])
            vi_after_30_min.append(vi)
        vi_s_30mins.append(vi_after_30_min)
        j = 0
        time_counter += 1
        print time_counter*30, 'minutes done bigM_max=', superMax

      if i>0 and i % 12 == 0:
        # every 12 corrections == 1 minute
        j += 1
        # print 'minutes', j
      i+=1


      superMax = -np.inf
      superL = -1
      superN = -1
      superSlice = -1

      #
      for slice in range(bigM.shape[0]):
          max_in_slice = bigM[slice].max()
          largest_indices = np.where(bigM[slice]==max_in_slice)
          # print largest_indices
          if max_in_slice > superMax:
              
              # found a new large one
              l,n = largest_indices[0][0], largest_indices[1][0]
              superSlice = slice
              superL = l
              superN = n
              superMax = max_in_slice

              # print 'found', l, n, slice, max_in_slice
          
      if superMax < sureness_threshold:
        break



      # print 'merging', superL, superN, 'in slice ', superSlice, 'with', superMax

      image = volume[superSlice]
      prob = volume_prob[superSlice]
      # segmentation = volume_segmentation[slice]
      groundtruth = volume_groundtruth[superSlice]



      ### now we have a new max
      slice_with_max_value = np.array(out_volume[superSlice])

      rollback_slice_with_max_value = np.array(slice_with_max_value)

      last_vi = Util.vi(slice_with_max_value, groundtruth)

      # now we merge
      # print 'merging', superL, superN
      slice_with_max_value[slice_with_max_value == superN] = superL


    
      after_merge_vi = Util.vi(slice_with_max_value, groundtruth)
      # after_merge_ed = Util.vi(segmentation_copy, groundtruth)
      
      # pxlsize = len(np.where(before_segmentation_copy == l)[0])
      # pxlsize2 = len(np.where(before_segmentation_copy == n)[0])


      good_fix = False
      # print '-'*80
      # print 'vi diff', last_vi-after_merge_vi
      if after_merge_vi < last_vi:
        #
        # this is a good fix
        #
        good_fix = True
        good_fix_counter += 1
        # Util.view_labels(before_segmentation_copy,[l,n], crop=False)
        # print 'good fix'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2   
        fixes.append('Good')       
      else:
        #
        # this is a bad fix
        #
        good_fix = False
        bad_fix_counter += 1
        fixes.append('Bad')
        # print 'bad fix, excluding it..'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2





      # reset all l,n entries
      bigM[superSlice][superL,:] = -2
      bigM[superSlice][:, superL] = -2
      bigM[superSlice][superN,:] = -2
      bigM[superSlice][:, superN] = -2

      # re-calculate neighbors
      # grab new neighbors of l
      l_neighbors = Util.grab_neighbors(slice_with_max_value, superL)

      for l_neighbor in l_neighbors:
        # recalculate new neighbors of l

        if l_neighbor == 0:
            # ignore neighbor zero
            continue

        prediction = Patch.grab_group_test_and_unify(cnn, image, prob, slice_with_max_value, superL, l_neighbor, oversampling=oversampling)
        # print superL, l_neighbor
        # print 'new pred', prediction
        bigM[superSlice][superL,l_neighbor] = prediction
        bigM[superSlice][l_neighbor,superL] = prediction




      out_volume[superSlice] = slice_with_max_value

    return bigM, out_volume, fixes, vi_s_30mins


  @staticmethod
  def splits_global_from_M(cnn, bigM, volume, volume_prob, volume_segmentation, volume_groundtruth=np.zeros((1,1)), hours=.5, randomize=False, error_rate=0, oversampling=False):


    bigM = np.array(bigM)
    # for development, we just need the matrix and the patches
    # return bigM, None, global_patches

    out_volume = np.array(volume_segmentation)
    # return out_volume

    good_fix_counter = 0
    bad_fix_counter = 0
    # error_rate = 0
    fixes = []
    vi_s_30mins = []

    superMax = -np.inf
    j = 0 # minute counter
    # for i in range(60): # no. corrections in 1 minute
    #for i in range(17280): # no. corrections in 24 h
    if hours == -1:
      # no timelimit == 30 days
      hours = 24*30

    corrections_time_limit = int(hours * 60 * 12)
    time_counter = 0
    for i in range(corrections_time_limit):
    # while True: # no time limit
      # print 'Correction', i

      if (j>0 and j % 30 == 0):
        # compute VI every 30 minutes
        vi_after_30_min = []
        for ov in range(out_volume.shape[0]):
            vi = Util.vi(volume_groundtruth[ov], out_volume[ov])
            vi_after_30_min.append(vi)
        vi_s_30mins.append(vi_after_30_min)
        j = 0
        time_counter += 1
        print time_counter*30, 'minutes done bigM_max=', superMax

      if i>0 and i % 12 == 0:
        # every 12 corrections == 1 minute
        j += 1
        # print 'minutes', j

      superMax = -np.inf
      superL = -1
      superN = -1
      superSlice = -1

      #
      for slice in range(bigM.shape[0]):
          max_in_slice = bigM[slice].max()
          largest_indices = np.where(bigM[slice]==max_in_slice)
          # print largest_indices
          if max_in_slice > superMax:
              
              # found a new large one
              l,n = largest_indices[0][0], largest_indices[1][0]
              superSlice = slice
              superL = l
              superN = n
              superMax = max_in_slice

              # print 'found', l, n, slice, max_in_slice
          
      if superMax <= 0.:
        break

      if randomize:
        superMax = .5
        superSlice = np.random.choice(bigM.shape[0])

        uniqueIDs = np.where(bigM[superSlice] > -3)

        superL = np.random.choice(uniqueIDs[0])

        neighbors = Util.grab_neighbors(volume_segmentation[superSlice], superL)
        superN = np.random.choice(neighbors)


      # print 'merging', superL, superN, 'in slice ', superSlice, 'with', superMax

      image = volume[superSlice]
      prob = volume_prob[superSlice]
      # segmentation = volume_segmentation[slice]
      groundtruth = volume_groundtruth[superSlice]



      ### now we have a new max
      slice_with_max_value = np.array(out_volume[superSlice])

      rollback_slice_with_max_value = np.array(slice_with_max_value)

      last_vi = Util.vi(slice_with_max_value, groundtruth)

      # now we merge
      # print 'merging', superL, superN
      slice_with_max_value[slice_with_max_value == superN] = superL


    
      after_merge_vi = Util.vi(slice_with_max_value, groundtruth)
      # after_merge_ed = Util.vi(segmentation_copy, groundtruth)
      
      # pxlsize = len(np.where(before_segmentation_copy == l)[0])
      # pxlsize2 = len(np.where(before_segmentation_copy == n)[0])


      good_fix = False
      # print '-'*80
      # print 'vi diff', last_vi-after_merge_vi
      if after_merge_vi < last_vi:
        #
        # this is a good fix
        #
        good_fix = True
        good_fix_counter += 1
        # Util.view_labels(before_segmentation_copy,[l,n], crop=False)
        # print 'good fix'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2   
        fixes.append('Good')       
      else:
        #
        # this is a bad fix
        #
        good_fix = False
        bad_fix_counter += 1
        fixes.append('Bad')
        # print 'bad fix, excluding it..'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2

      #
      #
      # ERROR RATE
      #
      rnd = random.random()

      if rnd < error_rate:
        # no matter what, this is a user error
        good_fix = not good_fix
        print 'user err'
      



      # reset all l,n entries
      bigM[superSlice][superL,superN] = -2
      bigM[superSlice][superN,superL] = -2

      if good_fix:

        bigM[superSlice][superL,:] = -2
        bigM[superSlice][:, superL] = -2
        bigM[superSlice][superN,:] = -2
        bigM[superSlice][:, superN] = -2

        # re-calculate neighbors
        # grab new neighbors of l
        l_neighbors = Util.grab_neighbors(slice_with_max_value, superL)

        for l_neighbor in l_neighbors:
          # recalculate new neighbors of l

          if l_neighbor == 0:
              # ignore neighbor zero
              continue

          prediction = Patch.grab_group_test_and_unify(cnn, image, prob, slice_with_max_value, superL, l_neighbor, oversampling=oversampling)
          # print superL, l_neighbor
          # print 'new pred', prediction
          bigM[superSlice][superL,l_neighbor] = prediction
          bigM[superSlice][l_neighbor,superL] = prediction

      else:

        slice_with_max_value = rollback_slice_with_max_value




      out_volume[superSlice] = slice_with_max_value

    return bigM, out_volume, fixes, vi_s_30mins


  @staticmethod
  def splits_global(cnn, volume, volume_prob, volume_segmentation, volume_groundtruth=np.zeros((1,1)), randomize=False, error_rate=0, sureness_threshold=0., smallest_first=False, oversampling=False, verbose=True, max=10000):

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
    # for development, we just need the matrix and the patches
    # return bigM, None, global_patches

    out_volume = np.array(volume_segmentation)
    # return out_volume

    good_fix_counter = 0
    bad_fix_counter = 0
    # error_rate = 0

    for i in range(360):
    # while (matrix.max() >= .5):

      # print i

      superMax = -np.inf
      superL = -1
      superN = -1
      superSlice = -1

      #
      for slice in range(bigM.shape[0]):
          max_in_slice = bigM[slice].max()
          largest_indices = np.where(bigM[slice]==max_in_slice)
          # print largest_indices
          if max_in_slice > superMax:
              
              # found a new large one
              l,n = largest_indices[0][0], largest_indices[1][0]
              superSlice = slice
              superL = l
              superN = n
              superMax = max_in_slice

              # print 'found', l, n, slice, max_in_slice
          
      if randomize:
        superMax = .5
        superSlice = np.random.choice(bigM.shape[0])

        uniqueIDs = np.where(bigM[superSlice] > -3)

        superL = np.random.choice(uniqueIDs[0])

        neighbors = Util.grab_neighbors(volume_segmentation[superSlice], superL)
        superN = np.random.choice(neighbors)


      # print 'merging', superL, superN, 'in slice ', superSlice, 'with', superMax

      image = volume[superSlice]
      prob = volume_prob[superSlice]
      # segmentation = volume_segmentation[slice]
      groundtruth = volume_groundtruth[superSlice]



      ### now we have a new max
      slice_with_max_value = np.array(out_volume[superSlice])

      rollback_slice_with_max_value = np.array(slice_with_max_value)

      last_vi = Util.vi(slice_with_max_value, groundtruth)

      # now we merge
      # print 'merging', superL, superN
      slice_with_max_value[slice_with_max_value == superN] = superL


    
      after_merge_vi = Util.vi(slice_with_max_value, groundtruth)
      # after_merge_ed = Util.vi(segmentation_copy, groundtruth)
      
      # pxlsize = len(np.where(before_segmentation_copy == l)[0])
      # pxlsize2 = len(np.where(before_segmentation_copy == n)[0])


      good_fix = False
      # print '-'*80
      # print 'vi diff', last_vi-after_merge_vi
      if after_merge_vi < last_vi:
        #
        # this is a good fix
        #
        good_fix = True
        good_fix_counter += 1
        # Util.view_labels(before_segmentation_copy,[l,n], crop=False)
        # print 'good fix'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2          
      else:
        #
        # this is a bad fix
        #
        good_fix = False
        bad_fix_counter += 1
        # print 'bad fix, excluding it..'
        # print 'size first label:', pxlsize
        # print 'size second label:',pxlsize2

      #
      #
      # ERROR RATE
      #
      rnd = random.random()

      if rnd < error_rate:
        # no matter what, this is a user error
        good_fix = not good_fix
        print 'user err'
      



      # reset all l,n entries
      bigM[superSlice][superL,superN] = -2
      bigM[superSlice][superN,superL] = -2

      if good_fix:


        # re-calculate neighbors
        # grab new neighbors of l
        l_neighbors = Util.grab_neighbors(slice_with_max_value, superL)

        for l_neighbor in l_neighbors:
          # recalculate new neighbors of l

          if l_neighbor == 0:
              # ignore neighbor zero
              continue

          prediction = Patch.grab_group_test_and_unify(cnn, image, prob, slice_with_max_value, superL, l_neighbor, oversampling=oversampling)
          # print superL, l_neighbor
          # print 'new pred', prediction
          bigM[superSlice][superL,l_neighbor] = prediction
          bigM[superSlice][l_neighbor,superL] = prediction

      else:

        slice_with_max_value = rollback_slice_with_max_value




      out_volume[superSlice] = slice_with_max_value

    return bigM, out_volume, global_patches









  @staticmethod
  def splits_user_simulated(cnn, image, prob, segmentation, groundtruth=np.zeros((1,1)), error_rate=0, sureness_threshold=0., smallest_first=False, oversampling=False, verbose=True, max=10000):
    '''
    '''
    t0 = time.time()
    patches = Patch.patchify(image, prob, segmentation, oversampling=oversampling, max=max)
    if verbose:
      print len(patches), 'generated in', time.time()-t0, 'seconds.'

    t0 = time.time()
    grouped_patches = Patch.group(patches)
    if verbose:
      print 'Grouped into', len(grouped_patches.keys()), 'patches in', time.time()-t0, 'seconds.'


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
      

    #
    # NOW the matrix is filled and we can start merging
    #

    print 'Start M', len(np.where(M!=-1)[0])

    # sureness_threshold = 0.
    matrix = np.array(M)
    segmentation_copy = np.array(segmentation)


    before_vi = Util.vi(segmentation_copy, groundtruth)

    # we keep track of the following values
    vi_s = []
    ed_s = []
    merge_pairs = []
    surenesses = []

    last_vi = before_vi

    good_fix_counter = 0
    bad_fix_counter = 0

    # now the loop
    t0 = time.time()
    while (matrix.max() >= sureness_threshold):
        


        sureness = matrix.max()
        
        largest_indices = np.where(matrix==sureness)


        picked = 0
        l,n = largest_indices[0][picked], largest_indices[1][picked]
        

        # print 'M', len(np.where(np.logical_and(matrix!=-1, matrix!=-2))[0])




        before_segmentation_copy = np.array(segmentation_copy)
        segmentation_copy[segmentation_copy == n] = l


        
 
      
        after_merge_vi = Util.vi(segmentation_copy, groundtruth)
        # after_merge_ed = Util.vi(segmentation_copy, groundtruth)
        
        pxlsize = len(np.where(before_segmentation_copy == l)[0])
        pxlsize2 = len(np.where(before_segmentation_copy == n)[0])


        good_fix = False
        # print '-'*80
        # print 'vi diff', last_vi-after_merge_vi
        if after_merge_vi < last_vi:
          #
          # this is a good fix
          #
          good_fix = True
          good_fix_counter += 1
          # Util.view_labels(before_segmentation_copy,[l,n], crop=False)
          # print 'good fix'
          # print 'size first label:', pxlsize
          # print 'size second label:',pxlsize2          
        else:
          #
          # this is a bad fix
          #
          good_fix = False
          bad_fix_counter += 1
          # print 'bad fix, excluding it..'
          # print 'size first label:', pxlsize
          # print 'size second label:',pxlsize2

        #
        #
        # ERROR RATE
        #
        rnd = random.random()
        if rnd < error_rate:
          # no matter what, this is a user error
          good_fix = not good_fix
          print 'user err'

        matrix[l,n] = -2
        matrix[n,l] = -2

        if good_fix:
          #
          # PERFORM THE MERGE
          #
          #

          # reset all l,n entries
          matrix[l,:] = -2
          matrix[:,l] = -2
          matrix[n,:] = -2
          matrix[:,n] = -2

          vi_s.append(after_merge_vi)
          surenesses.append(sureness)

          merge_pairs.append((l,n))
          
          # grab new neighbors of l
          l_neighbors = Util.grab_neighbors(segmentation_copy, l)

          for l_neighbor in l_neighbors:
              # recalculate new neighbors of l
              
              if l_neighbor == 0:
                  # ignore neighbor zero
                  continue
          
              prediction = Patch.grab_group_test_and_unify(cnn, image, prob, segmentation_copy, l, l_neighbor, oversampling=oversampling)
          
              matrix[l,l_neighbor] = prediction
              matrix[l_neighbor,l] = prediction


          last_vi = Util.vi(segmentation_copy, groundtruth)

        if not good_fix:
          #
          # DO NOT PERFORM THIS MERGE
          #
          segmentation_copy = before_segmentation_copy


        if bad_fix_counter + good_fix_counter == 12:
          print 'ALL DONE - LIMIT REACHED'
          break


    if verbose:
      print 'Merge loop finished in', time.time()-t0, 'seconds.'

    if groundtruth.shape[0]>1:
      min_vi_index = vi_s.index(np.min(vi_s))
      if verbose:
        print 'Before VI:', before_vi
        print 'Smallest VI:', vi_s[min_vi_index]
        print 'Sureness threshold:', surenesses[min_vi_index]


    return vi_s, merge_pairs, surenesses, good_fix_counter, bad_fix_counter



  @staticmethod
  def splits(cnn, image, prob, segmentation, groundtruth=np.zeros((1,1)), sureness_threshold=0., smallest_first=False, oversampling=False, verbose=True, max=10000):
    '''
    '''
    t0 = time.time()
    patches = Patch.patchify(image, prob, segmentation, oversampling=oversampling, max=max)
    if verbose:
      print len(patches), 'generated in', time.time()-t0, 'seconds.'

    t0 = time.time()
    grouped_patches = Patch.group(patches)
    if verbose:
      print 'Grouped into', len(grouped_patches.keys()), 'patches in', time.time()-t0, 'seconds.'


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
      

    #
    # NOW the matrix is filled and we can start merging
    #

    # sureness_threshold = 0.
    matrix = np.array(M)
    segmentation_copy = np.array(segmentation)

    if groundtruth.shape[0]>1:
      before_vi = Util.vi(segmentation_copy, groundtruth)

    # we keep track of the following values
    vi_s = []
    ed_s = []
    merge_pairs = []
    surenesses = []

    # now the loop
    t0 = time.time()
    while (matrix.max() >= sureness_threshold):
        
        sureness = matrix.max()
        
        largest_indices = np.where(matrix==sureness)
        #
        #
        # TO TRY: merge smaller ones with smaller ones first
        #



        picked = 0

        if smallest_first:
          smallest = np.Infinity
          smallest_label = -1

          for i,label in enumerate(largest_indices[0]):
            current_size = len(segmentation_copy[segmentation_copy == label])
            if current_size < smallest:
              smallest = current_size
              smallest_label = label
              picked = i


        l,n = largest_indices[0][picked], largest_indices[1][picked]
        

        #
        # TODO also check for alternative n's
        #






        segmentation_copy[segmentation_copy == n] = l
        
        # reset all l,n entries
        matrix[l,:] = -2
        matrix[:,l] = -2
        matrix[n,:] = -2
        matrix[:,n] = -2
        
        if groundtruth.shape[0]>1:
          after_merge_vi = Util.vi(segmentation_copy, groundtruth)
          # after_merge_ed = Util.vi(segmentation_copy, groundtruth)
          vi_s.append(after_merge_vi)

        merge_pairs.append((l,n))
        surenesses.append(sureness)
        
        
        # grab new neighbors of l
        l_neighbors = Util.grab_neighbors(segmentation_copy, l)

        for l_neighbor in l_neighbors:
            # recalculate new neighbors of l
            
            if l_neighbor == 0:
                # ignore neighbor zero
                continue
        
            prediction = Patch.grab_group_test_and_unify(cnn, image, prob, segmentation_copy, l, l_neighbor, oversampling=oversampling)
        
            matrix[l,l_neighbor] = prediction
            matrix[l_neighbor,l] = prediction

    if verbose:
      print 'Merge loop finished in', time.time()-t0, 'seconds.'

    if groundtruth.shape[0]>1:
      min_vi_index = vi_s.index(np.min(vi_s))
      if verbose:
        print 'Before VI:', before_vi
        print 'Smallest VI:', vi_s[min_vi_index]
        print 'Sureness threshold:', surenesses[min_vi_index]


    return vi_s, merge_pairs, surenesses



  @staticmethod
  def fix_single_merge(cnn, cropped_image, cropped_prob, cropped_binary, N=10, invert=True, dilate=True, 
                       border_seeds=True, erode=False, debug=False, before_merge_error=None,
                       real_border=np.zeros((1,1)), oversampling=False, crop=True):
    '''
    invert: True/False for invert or gradient image
    '''

    bbox = mh.bbox(cropped_binary)

    orig_cropped_image = np.array(cropped_image)
    orig_cropped_prob  = np.array(cropped_prob)
    orig_cropped_binary = np.array(cropped_binary)



    speed_image = None
    if invert:
      speed_image = Util.invert(cropped_image, smooth=True, sigma=2.5)
    else:
      speed_image = Util.gradient(cropped_image)


    dilated_binary = np.array(cropped_binary, dtype=np.bool)
    if dilate:
      for i in range(20):
          dilated_binary = mh.dilate(dilated_binary)      

    # Util.view(dilated_binary, large=True)

    borders = np.zeros(cropped_binary.shape)

    best_border_prediction = np.inf
    best_border_image = np.zeros(cropped_binary.shape)

    original_border = mh.labeled.border(cropped_binary, 1, 0, Bc=mh.disk(3))

    results_no_border = []
    predictions = []
    borders = []
    results = []

    for n in range(N):
        ws = Util.random_watershed(dilated_binary, speed_image, border_seeds=border_seeds, erode=erode)
        
        if ws.max() == 0:
          continue

        ws_label1 = ws.max()
        ws_label2 = ws.max()-1
        border = mh.labeled.border(ws, ws_label1, ws_label2)

        # Util.view(ws, large=True)


        # Util.view(border, large=True)

        # print i, len(border[border==True])

        #
        # remove parts of the border which overlap with the original border
        #

        

        ws[cropped_binary == 0] = 0

        # Util.view(ws, large=False, color=False)        

        ws_label1_array = Util.threshold(ws, ws_label1)
        ws_label2_array = Util.threshold(ws, ws_label2)

        eroded_ws1 = np.array(ws_label1_array, dtype=np.bool)
        eroded_ws2 = np.array(ws_label2_array, dtype=np.bool)
        if erode:

          for i in range(5):
            eroded_ws1 = mh.erode(eroded_ws1)

          # Util.view(eroded_ws, large=True, color=False)

          dilated_ws1 = np.array(eroded_ws1)
          for i in range(5):
            dilated_ws1 = mh.dilate(dilated_ws1)


          for i in range(5):
            eroded_ws2 = mh.erode(eroded_ws2)

          # Util.view(eroded_ws, large=True, color=False)

          dilated_ws2 = np.array(eroded_ws2)
          for i in range(5):
            dilated_ws2 = mh.dilate(dilated_ws2)




          new_ws = np.zeros(ws.shape, dtype=np.uint8)
          new_ws[dilated_ws1 == 1] = ws_label1
          new_ws[dilated_ws2 == 1] = ws_label2


          ws = new_ws

          # Util.view(new_ws, large=True, color=True)

        # ws[original_border == 1] = 0
        
        prediction = Patch.grab_group_test_and_unify(cnn, cropped_image, cropped_prob, ws, ws_label1, ws_label2, oversampling=oversampling)
        
        if prediction == -1 or prediction >= .5:
          # invalid
          continue


        # here we have for one border
        # the border
        # the prediction
        # borders.append(border)
        # predictions.append(prediction)
        results.append((prediction, border))



    return results





    #     # if (prediction < best_border_prediction):
    #     #   best_border_prediction = prediction
    #     #   best_border_image = border
    #     #   print 'new best', n, prediction

    #     best_border_image = border

    #     borders += (border*prediction)

    #     result = np.array(cropped_binary)
    #     best_border_image[result==0] = 0
    #     result[best_border_image==1] = 2

    #     result = skimage.measure.label(result)

    #     result_no_border = np.array(result)
    #     result_no_border[best_border_image==1] = 0        

    #     predictions.append(prediction)
    #     results_no_border.append(result_no_border)

    #     real_borders.append(border)


    # # result = np.array(cropped_binary)
    # # best_border_image[result==0] = 0
    # # result[best_border_image==1] = 2

    # # result = skimage.measure.label(result)

    # # result_no_border = np.array(result)
    # # result_no_border[best_border_image==1] = 0
    # # result_no_border = mh.croptobbox(result_no_border)

    # # if before_merge_error == None:
    # #   continue        

    # # print result_no_border.shape, before_merge_error.shape


    # # if before_merge_error.shape[0] != result_no_border.shape[0] or before_merge_error.shape[1] != result_no_border.shape[1]:
    # #   result_no_border = np.resize(result_no_border, before_merge_error.shape)

    # # print 'vi', Util.vi(before_merge_error.astype(np.uint8), result_no_border.astype(np.uint8))

        
    # #     if debug:
    # #       Util.view(ws, text=str(i) + ' ' + str(prediction))
        

    # result = np.array(cropped_binary)
    # best_border_image[result==0] = 0
    # result[best_border_image==1] = 2

    # result = skimage.measure.label(result)

    # result_no_border = np.array(result)
    # result_no_border[best_border_image==1] = 0


    # return borders, best_border_image, result, result_no_border, results_no_border, predictions





















  @staticmethod
  def splits_nn(cnn, image, prob, segmentation, groundtruth=np.zeros((1,1)), smallest_first=False, oversampling=False, verbose=True):
    '''
    '''
    t0 = time.time()
    patches = Patch.patchify(image, prob, segmentation, oversampling=oversampling)
    if verbose:
      print len(patches), 'generated in', time.time()-t0, 'seconds.'

    t0 = time.time()
    grouped_patches = Patch.group(patches)
    if verbose:
      print 'Grouped into', len(grouped_patches.keys()), 'patches in', time.time()-t0, 'seconds.'


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
      

    #
    # NOW the matrix is filled and we can start merging
    #

    sureness_threshold = 0.
    matrix = np.array(M)
    segmentation_copy = np.array(segmentation)

    if groundtruth.shape[0]>1:
      before_vi = Util.vi(segmentation_copy, groundtruth)

    # we keep track of the following values
    vi_s = []
    merge_pairs = []
    surenesses = []

    # now the loop
    t0 = time.time()
    while (matrix.max() >= sureness_threshold):
        
        sureness = matrix.max()
        
        largest_indices = np.where(matrix==sureness)
        #
        #
        # TO TRY: merge smaller ones with smaller ones first
        #



        picked = 0

        if smallest_first:
          smallest = np.Infinity
          smallest_label = -1

          for i,label in enumerate(largest_indices[0]):
            current_size = len(segmentation_copy[segmentation_copy == label])
            if current_size < smallest:
              smallest = current_size
              smallest_label = label
              picked = i


        l,n = largest_indices[0][picked], largest_indices[1][picked]
        

        #
        # TODO also check for alternative n's
        #


        #
        #
        # NO NEIGHBOR UPDATES!!!
        #



        segmentation_copy[segmentation_copy == n] = l
        
        # reset all l,n entries
        matrix[l,n] = -2
        matrix[n,l] = -2
        matrix[n,:] = -2
        matrix[:,n] = -2
        
        if groundtruth.shape[0]>1:
          after_merge_vi = Util.vi(segmentation_copy, groundtruth)
          vi_s.append(after_merge_vi)

        merge_pairs.append((l,n))
        surenesses.append(sureness)
        
        
        # # grab new neighbors of l
        # l_neighbors = Util.grab_neighbors(segmentation_copy, l)

        # for l_neighbor in l_neighbors:
        #     # recalculate new neighbors of l
            
        #     if l_neighbor == 0:
        #         # ignore neighbor zero
        #         continue
        
        #     prediction = Patch.grab_group_test_and_unify(cnn, image, prob, segmentation_copy, l, l_neighbor, oversampling=oversampling)
        
        #     matrix[l,l_neighbor] = prediction
        #     matrix[l_neighbor,l] = prediction

    if verbose:
      print 'Merge loop finished in', time.time()-t0, 'seconds.'

    if groundtruth.shape[0]>1:
      min_vi_index = vi_s.index(np.min(vi_s))
      if verbose:
        print 'Before VI:', before_vi
        print 'Smallest VI:', vi_s[min_vi_index]
        print 'Sureness threshold:', surenesses[min_vi_index]


    return vi_s, merge_pairs, surenesses