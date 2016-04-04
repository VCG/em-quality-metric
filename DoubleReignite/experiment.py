import numpy as np
import mahotas as mh

from fixer import Fixer
from uglify import Uglify
from util import Util

class Experiment(object):


  @staticmethod
  def run_split_test(cnn, 
                     bbox=[0,1024,0,1024],
                     data='gt',
                     slices=[70,71,72,73,74],
                     oversampling=False,
                     smallest_first=False,
                     no_splits=2,
                     keep_zeros=True,
                     fill_zeros=False,
                     verbose=False):
    '''
    data: gt/rhoana
    slices: [x,y,z..]
    oversampling: True/False
    smallest_first: True/False
    no_splits: 1/2/3..
    '''

    print '-'*80    
    print '-'*80    
    print 'New Experiment:'
    print '  Data:', data
    print '  Slices:', slices
    print '  Oversampling:', oversampling
    print '  Merge smallest first:', smallest_first
    print '  No. splits to uglify:', no_splits
    print '  Keep zeros in segmentation:', keep_zeros

    global_eds = []
    global_ris = []
    global_vis = []
    global_vi_diffs = []
    global_ed_diffs = []
    global_surenesses = []
    global_ed_surenesses = []
    global_merge_pairs = []
    global_ugly_segmentations = []
    global_best_indices = []

    for s in slices:

      if verbose:
        print '-'*80    
        print 'Working on slice',s

      # load slice
      input_image, input_prob, input_gold, input_rhoana = Util.read_section(s, keep_zeros=keep_zeros, fill_zeros=fill_zeros)

      # apply bbox
      input_image = input_image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
      input_prob = input_prob[bbox[0]:bbox[1], bbox[2]:bbox[3]]
      input_gold = input_gold[bbox[0]:bbox[1], bbox[2]:bbox[3]]
      input_rhoana = input_rhoana[bbox[0]:bbox[1], bbox[2]:bbox[3]]

      # choose segmentation based on data mode
      framed_gold = Util.frame_image(input_gold)
      framed_rhoana = Util.frame_image(input_rhoana)

      if data=='gt':
        segmentation = framed_gold

        # if GT, uglify the segmenation based on the number of splits
        ugly_segmentation = Uglify.split(input_image, input_prob, segmentation, n=no_splits)

      else:
        segmentation = framed_gold
        ugly_segmentation = framed_rhoana

      global_ugly_segmentations.append(ugly_segmentation)

      before_vi = Util.vi(ugly_segmentation, segmentation)
      # before_ri = Util.ri(ugly_segmentation, segmentation)
      before_ed = Util.ed(ugly_segmentation, segmentation)

      if verbose:
        print 'Labels before:', len(Util.get_histogram(segmentation))
        print 'Labels after:', len(Util.get_histogram(ugly_segmentation))
      
        print 'VI after uglifying:', before_vi


      #
      # now run the fixer
      #
      vi_s, ed_s, merge_pairs, surenesses = Fixer.splits(cnn, 
                                                   input_image,
                                                   input_prob,
                                                   ugly_segmentation,
                                                   segmentation,
                                                   smallest_first=smallest_first,
                                                   oversampling=oversampling,
                                                   verbose=verbose)

      best_index = vi_s.index(np.min(vi_s))
      best_vi = vi_s[best_index]
      best_sureness = surenesses[best_index]

      best_ed_index = ed_s.index(np.min(ed_s))
      best_ed = ed_s[best_ed]
      best_sureness_ed = surenesses[best_ed]

      vi_diff = before_vi - best_vi
      ed_diff = before_ed - best_ed

      global_vis.append(best_vi)
      global_vi_diffs.append(vi_diff)
      global_surenesses.append(best_sureness)
      global_merge_pairs.append(merge_pairs)
      global_best_indices.append(best_index)

      global_eds.append(best_ed)
      global_ed_diffs.append(ed_diff)
      global_surenesses_ed.append(best_sureness_ed)      

    #
    # now all done
    #

    print 'VI:'
    Util.stats(global_vis)

    print 'VI before-after:'
    Util.stats(global_vi_diffs)

    print 'Surenesses:'
    Util.stats(global_surenesses)

    print 'ED:'
    Util.stats(global_eds)    

    print 'ED before-after:'
    Util.stats(global_ed_diffs)    

    print 'ED Surenesses:'
    Util.stats(global_surenesses_ed)    

    return global_vis, global_vi_diffs, global_surenesses, global_eds, global_ed_diffs, global_surenesses_ed, global_merge_pairs, global_best_indices, global_ugly_segmentations



  # @staticmethod
  # def run_split_test_nn(cnn, 
  #                    bbox=[0,1024,0,1024],
  #                    data='gt',
  #                    slices=[70,71,72,73,74],
  #                    oversampling=False,
  #                    smallest_first=False,
  #                    no_splits=2,
  #                    keep_zeros=True,
  #                    verbose=False):
  #   '''
  #   data: gt/rhoana
  #   slices: [x,y,z..]
  #   oversampling: True/False
  #   smallest_first: True/False
  #   no_splits: 1/2/3..
  #   '''

  #   print '-'*80    
  #   print '-'*80    
  #   print 'New Experiment:'
  #   print '  Data:', data
  #   print '  Slices:', slices
  #   print '  Oversampling:', oversampling
  #   print '  Merge smallest first:', smallest_first
  #   print '  No. splits to uglify:', no_splits
  #   print '  Keep zeros in segmentation:', keep_zeros


  #   global_vis = []
  #   global_vi_diffs = []
  #   global_surenesses = []
  #   global_merge_pairs = []

  #   for s in slices:

  #     if verbose:
  #       print '-'*80    
  #       print 'Working on slice',s

  #     # load slice
  #     input_image, input_prob, input_gold, input_rhoana = Util.read_section(s, keep_zeros=keep_zeros)

  #     # apply bbox
  #     input_image = input_image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
  #     input_prob = input_prob[bbox[0]:bbox[1], bbox[2]:bbox[3]]
  #     input_gold = input_gold[bbox[0]:bbox[1], bbox[2]:bbox[3]]
  #     input_rhoana = input_rhoana[bbox[0]:bbox[1], bbox[2]:bbox[3]]

  #     # choose segmentation based on data mode
  #     framed_gold = Util.frame_image(input_gold)
  #     framed_rhoana = Util.frame_image(input_rhoana)

  #     if data=='gt':
  #       segmentation = framed_gold

  #       # if GT, uglify the segmenation based on the number of splits
  #       ugly_segmentation = Uglify.split(input_image, input_prob, segmentation, n=no_splits)

  #     else:
  #       segmentation = framed_rhoana
  #       ugly_segmentation = segmentation

  #     before_vi = Util.vi(ugly_segmentation, segmentation)

  #     if verbose:
  #       print 'Labels before:', len(Util.get_histogram(segmentation))
  #       print 'Labels after:', len(Util.get_histogram(ugly_segmentation))
      
  #       print 'VI after uglifying:', before_vi


  #     #
  #     # now run the fixer
  #     #
  #     vi_s, merge_pairs, surenesses = Fixer.splits_nn(cnn, 
  #                                                  input_image,
  #                                                  input_prob,
  #                                                  ugly_segmentation,
  #                                                  framed_gold,
  #                                                  smallest_first=smallest_first,
  #                                                  oversampling=oversampling,
  #                                                  verbose=verbose)

  #     best_index = vi_s.index(np.min(vi_s))
  #     best_vi = vi_s[best_index]
  #     best_sureness = surenesses[best_index]

  #     vi_diff = before_vi - best_vi

  #     global_vis.append(best_vi)
  #     global_vi_diffs.append(vi_diff)
  #     global_surenesses.append(best_sureness)
  #     global_merge_pairs.append(merge_pairs)

  #   #
  #   # now all done
  #   #

  #   print 'VI:'
  #   Util.stats(global_vis)

  #   print 'VI before-after:'
  #   Util.stats(global_vi_diffs)

  #   print 'Surenesses:'
  #   Util.stats(global_surenesses)

  #   return global_vis, global_vi_diffs, global_surenesses, global_merge_pairs




  @staticmethod
  def run_merge_test(cnn, 
                     bbox=[0,1024,0,1024],
                     data='gt',
                     slices=[70,71,72,73,74],
                     min_pixels=1000,
                     N=20,
                     oversampling=False,
                     keep_zeros=True,
                     verbose=False):


    if data == 'rhoana':
      print 'not implemented yet'
      return [],[]

    print '-'*80    
    print '-'*80    
    print 'New Experiment:'
    print '  Data:', data
    print '  Slices:', slices
    print '  No. splits for borders:', N
    print '  Keep zeros in segmentation:', keep_zeros


    # global_vis = []
    global_vi_diffs = []
    # global_surenesses = []
    # global_merge_pairs = []

    for s in slices:

      if verbose:
        print '-'*80    
        print 'Working on slice',s

      # load slice
      input_image, input_prob, input_gold, input_rhoana = Util.read_section(s, keep_zeros=keep_zeros)

      # apply bbox
      input_image = input_image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
      input_prob = input_prob[bbox[0]:bbox[1], bbox[2]:bbox[3]]
      input_gold = input_gold[bbox[0]:bbox[1], bbox[2]:bbox[3]]
      input_rhoana = input_rhoana[bbox[0]:bbox[1], bbox[2]:bbox[3]]

      framed_gold = Util.frame_image(input_gold, shape=(200,200))

      hist = Util.get_histogram(framed_gold.astype(np.uint64))
      labels = range(len(hist))
      np.random.shuffle(labels)


      slice_vi_diffs = []


      for l in labels:
          
          if l == 0:
              continue

          if len(framed_gold[framed_gold == l]) < min_pixels:
              continue    

          neighbors = Util.grab_neighbors(framed_gold, l)
          np.random.shuffle(neighbors)

          good_neighbors = []
          
          for n in neighbors:

              if n == 0:
                  continue

              if len(framed_gold[framed_gold == n]) < min_pixels:
                  continue
                  
              good_neighbors.append(n)
          
          if len(good_neighbors) > 0:
              
              for n in good_neighbors:
                  
                  # print 'merging', l, n


                  before_merge_error = np.zeros(framed_gold.shape)
                  before_merge_error[framed_gold == l] = 1
                  before_merge_error[framed_gold == n] = 2
                  before_merge_error = mh.croptobbox(before_merge_error)
                  

                  cropped_image, cropped_prob, cropped_segmentation, cropped_binary, bbox, real_border = Uglify.merge_label(input_image,
                                                                                                             input_prob,
                                                                                                             framed_gold,
                                                                                                             l,
                                                                                                             n,
                                                                                                             crop=True)

                  
                  vi_before = Util.vi(before_merge_error[10:-10,10:-10].astype(np.uint8), mh.croptobbox(cropped_binary)[10:-10,10:-10].astype(np.uint8))
                  
                  # print 'VI after merge error:', vi_before
                  
                  borders, best_border_image, result, result_no_border, results_no_border, predictions = Fixer.fix_single_merge(cnn, cropped_image, cropped_prob,
                                                                                                cropped_binary, real_border=real_border, N=N, erode=True, oversampling=False)

                  if result_no_border.shape[0] == 0:
                      continue

                  if best_border_image.max() == 0:
                      # print 'no solution'
                      continue
                      
      #             if before_merge_error.shape[0] != result_no_border.shape[0] or before_merge_error.shape[1] != result_no_border.shape[1]:
      #               result_no_border = np.resize(result_no_border, before_merge_error.shape)            
                  
      #             if before_merge_error.size != mh.croptobbox(r)

      #             compare_result = np.zeros(before_merge_error.shape, dtype=np.uint8)
      #             compare_result[:] = result_no_border[101:-101, 101:-101][0:before_merge_error.shape[0], 0:before_merge_error.shape[1]]

                  best_vi = np.inf
                  vi_diffs = []

                  # sorted_predictions = sorted(predictions)

                  for r in results_no_border:

                      if r.shape[0] == 0:
                          continue
                          
                      r = r[100:-100, 100:-100]


                      result_no_border_center = (r.shape[0]/2, r.shape[1]/2)
                      before_merge_center = (before_merge_error.shape[0]/2-10, before_merge_error.shape[1]/2-10)



                      r = r[result_no_border_center[0]-before_merge_center[0]:result_no_border_center[0]+before_merge_center[0],
                                                          result_no_border_center[1]-before_merge_center[1]:result_no_border_center[1]+before_merge_center[1]]

                      b = before_merge_error[result_no_border_center[0]-before_merge_center[0]:result_no_border_center[0]+before_merge_center[0],
                                                              result_no_border_center[1]-before_merge_center[1]:result_no_border_center[1]+before_merge_center[1]]

                      vi_after_fixing = Util.vi(b.astype(np.uint8), r.astype(np.uint8))

                      vi_diffs.append(vi_before-vi_after_fixing)



                  # now we have vi_diffs for this one merge error
                  slice_vi_diffs.append((vi_diffs, predictions))




      global_vi_diffs.append(slice_vi_diffs)


    vi_correction_bins = [0,0,0,0,0]
    bin_counts = [0,0,0,0,0]

    for s in global_vi_diffs:
        for merge_errors in s:
            vi_diff_per_error = merge_errors[0]
            predictions_per_error = merge_errors[1]
            
            # sort by prediction
            found_borders = sorted(zip(vi_diff_per_error, predictions_per_error), key=lambda x: x[1])
            for i in range(5):
                if len(found_borders) > i:
                    for j in range(i,len(found_borders)):
                      print i,j,len(vi_correction_bins), len(found_borders)
                      vi_correction_bins[j] += found_borders[j][0]    
                      bin_counts[j] += 1
                    

    for i in range(5):
        vi_correction_bins[i] /= bin_counts[i]


    return global_vi_diffs, vi_correction_bins




















                      # if vi_after_fixing < best_vi:
                      #     best_vi = vi_after_fixing
              
              
              
                  # result_no_border = result_no_border[100:-100, 100:-100]

          
                  # result_no_border_center = (result_no_border.shape[0]/2, result_no_border.shape[1]/2)
                  # before_merge_center = (before_merge_error.shape[0]/2-10, before_merge_error.shape[1]/2-10)
                  
                  
                  
                  # result_no_border = result_no_border[result_no_border_center[0]-before_merge_center[0]:result_no_border_center[0]+before_merge_center[0],
                  #                                     result_no_border_center[1]-before_merge_center[1]:result_no_border_center[1]+before_merge_center[1]]
          
                  # before_merge_error = before_merge_error[result_no_border_center[0]-before_merge_center[0]:result_no_border_center[0]+before_merge_center[0],
                  #                                         result_no_border_center[1]-before_merge_center[1]:result_no_border_center[1]+before_merge_center[1]]

                  
          
          
      #             shape_diff = map(int.__sub__, before_merge_error.shape, result_no_border.shape)
          
      #             if shape_diff[0] > 0:
      #                 result_no_border = np.pad(result_no_border, (shape_diff[0], 0), mode='minimum')
                  
                  
      #             or shape_diff[1] > 0:
      #                 # result_no_border is smaller
      #                 result_no_border = np.pad(result_no_border, (shape_diff[0], shape_diff[1]/2.), mode='minimum')
      #             elif shape_diff[0] < 0 or shape_diff[1] < 0:
      #                 before_merge_error = np.pad(before_merge_error, (shape_diff[0], shape_diff[1]/2.), mode='minimum')

          
                  # vi_after_fixing = Util.vi(before_merge_error.astype(np.uint8), result_no_border.astype(np.uint8))
          
                  # if vi_after_fixing > best_vi:
                  #     vi_after_fixing = best_vi
                  #     print 'a previous one was better'
                  #     previous_vi_better += 1
          
                  # vi_diffs2.append(vi_before-vi_after_fixing)
          
                  # print 'VI after fixing:', vi_after_fixing
              
                  # if vi_after_fixing < vi_before:
                  #     good += 1
                  # else:
                  #     bad += 1
              
                  # print '-'*80

