import numpy as np

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


    global_vis = []
    global_vi_diffs = []
    global_surenesses = []
    global_merge_pairs = []

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

      before_vi = Util.vi(ugly_segmentation, segmentation)

      if verbose:
        print 'Labels before:', len(Util.get_histogram(segmentation))
        print 'Labels after:', len(Util.get_histogram(ugly_segmentation))
      
        print 'VI after uglifying:', before_vi


      #
      # now run the fixer
      #
      vi_s, merge_pairs, surenesses = Fixer.splits(cnn, 
                                                   input_image,
                                                   input_prob,
                                                   ugly_segmentation,
                                                   framed_gold,
                                                   smallest_first=smallest_first,
                                                   oversampling=oversampling,
                                                   verbose=verbose)

      best_index = vi_s.index(np.min(vi_s))
      best_vi = vi_s[best_index]
      best_sureness = surenesses[best_index]

      vi_diff = before_vi - best_vi

      global_vis.append(best_vi)
      global_vi_diffs.append(vi_diff)
      global_surenesses.append(best_sureness)
      global_merge_pairs.append(merge_pairs)

    #
    # now all done
    #

    print 'VI:'
    Util.stats(global_vis)

    print 'VI before-after:'
    Util.stats(global_vi_diffs)

    print 'Surenesses:'
    Util.stats(global_surenesses)

    return global_vis, global_vi_diffs, global_surenesses, global_merge_pairs



  @staticmethod
  def run_split_test_nn(cnn, 
                     bbox=[0,1024,0,1024],
                     data='gt',
                     slices=[70,71,72,73,74],
                     oversampling=False,
                     smallest_first=False,
                     no_splits=2,
                     keep_zeros=True,
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


    global_vis = []
    global_vi_diffs = []
    global_surenesses = []
    global_merge_pairs = []

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

      # choose segmentation based on data mode
      framed_gold = Util.frame_image(input_gold)
      framed_rhoana = Util.frame_image(input_rhoana)

      if data=='gt':
        segmentation = framed_gold

        # if GT, uglify the segmenation based on the number of splits
        ugly_segmentation = Uglify.split(input_image, input_prob, segmentation, n=no_splits)

      else:
        segmentation = framed_rhoana
        ugly_segmentation = segmentation

      before_vi = Util.vi(ugly_segmentation, segmentation)

      if verbose:
        print 'Labels before:', len(Util.get_histogram(segmentation))
        print 'Labels after:', len(Util.get_histogram(ugly_segmentation))
      
        print 'VI after uglifying:', before_vi


      #
      # now run the fixer
      #
      vi_s, merge_pairs, surenesses = Fixer.splits_nn(cnn, 
                                                   input_image,
                                                   input_prob,
                                                   ugly_segmentation,
                                                   framed_gold,
                                                   smallest_first=smallest_first,
                                                   oversampling=oversampling,
                                                   verbose=verbose)

      best_index = vi_s.index(np.min(vi_s))
      best_vi = vi_s[best_index]
      best_sureness = surenesses[best_index]

      vi_diff = before_vi - best_vi

      global_vis.append(best_vi)
      global_vi_diffs.append(vi_diff)
      global_surenesses.append(best_sureness)
      global_merge_pairs.append(merge_pairs)

    #
    # now all done
    #

    print 'VI:'
    Util.stats(global_vis)

    print 'VI before-after:'
    Util.stats(global_vi_diffs)

    print 'Surenesses:'
    Util.stats(global_surenesses)

    return global_vis, global_vi_diffs, global_surenesses, global_merge_pairs
