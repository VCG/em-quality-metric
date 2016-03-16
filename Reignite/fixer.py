import time
import mahotas as mh
import numpy as np
import skimage.measure


from patch import Patch
from util import Util

class Fixer(object):




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
        
        if prediction == -1:
          # invalid
          continue

        # if (prediction < best_border_prediction):
        #   best_border_prediction = prediction
        #   best_border_image = border
        #   print 'new best', n, prediction

        best_border_image = border

        borders += (border*prediction)

        result = np.array(cropped_binary)
        best_border_image[result==0] = 0
        result[best_border_image==1] = 2

        result = skimage.measure.label(result)

        result_no_border = np.array(result)
        result_no_border[best_border_image==1] = 0        

        predictions.append(prediction)
        results_no_border.append(result_no_border)


    # result = np.array(cropped_binary)
    # best_border_image[result==0] = 0
    # result[best_border_image==1] = 2

    # result = skimage.measure.label(result)

    # result_no_border = np.array(result)
    # result_no_border[best_border_image==1] = 0
    # result_no_border = mh.croptobbox(result_no_border)

    # if before_merge_error == None:
    #   continue        

    # print result_no_border.shape, before_merge_error.shape


    # if before_merge_error.shape[0] != result_no_border.shape[0] or before_merge_error.shape[1] != result_no_border.shape[1]:
    #   result_no_border = np.resize(result_no_border, before_merge_error.shape)

    # print 'vi', Util.vi(before_merge_error.astype(np.uint8), result_no_border.astype(np.uint8))

        
    #     if debug:
    #       Util.view(ws, text=str(i) + ' ' + str(prediction))
        

    result = np.array(cropped_binary)
    best_border_image[result==0] = 0
    result[best_border_image==1] = 2

    result = skimage.measure.label(result)

    result_no_border = np.array(result)
    result_no_border[best_border_image==1] = 0


    return borders, best_border_image, result, result_no_border, results_no_border, predictions





















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