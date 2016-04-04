from cnn import CNN
from dyn_cnn import DynCNN
from experiment import Experiment
from util import Util
from patch import Patch
from fixer import Fixer
from uglify import Uglify

from PIL import Image
import mahotas as mh
import numpy as np
import os
import cPickle as pickle
from string import Template
import shutil

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

class Stats(object):

  @staticmethod
  def create(cnn_name, patch_path, inputs, trained_gt=True, cnn_class=CNN):


    OUTPUT_PATH = '/Volumes/DATA1/cnn_analysis/'
    if not os.path.isdir(OUTPUT_PATH):
      # this is cluster
      OUTPUT_PATH = '/n/regal/pfister_lab/haehn/cnn_analysis/'
    CNN_NAME = cnn_name
    REAL_OUTPUT_PATH = OUTPUT_PATH+CNN_NAME
    if not os.path.isdir(REAL_OUTPUT_PATH):
        os.makedirs(REAL_OUTPUT_PATH)
    shutil.copyfile(OUTPUT_PATH+'style.css', REAL_OUTPUT_PATH+os.sep+'style.css')
    PATCH_PATH = patch_path
    CNN_PATCHES = inputs
    EPOCH = 200
    TEST_LOSS = 0.01
    TEST_ACC = 0.99
    CONFIGURATION = 'dfdsf'
    SLICES = [70,72]#[70,71,72,73,74]
    SPLIT_BBOX = [0,1024,0,1024]#[200,500,200,500]
    MERGE_BBOX = [0,1024,0,1024]#[200,700,200,700]#[0,1024,0,1024]
    MERGE_N = 5
    MERGE_ITERATIONS = 1

    # load CNN
    NETWORK = cnn_class(CNN_NAME, PATCH_PATH, CNN_PATCHES)
    # get configuration, epoch, test_loss, test_acc
    EPOCH = NETWORK._epochs
    TEST_LOSS = NETWORK._test_loss
    TEST_ACC = NETWORK._test_acc
    CONFIGURATION = NETWORK._configuration


    patches_html = ''
    for img in CNN_PATCHES:
        img_path = REAL_OUTPUT_PATH+os.sep+img+'.png'
        shutil.copyfile(OUTPUT_PATH+'gfx/'+img+'.png', img_path)
        patches_html += '<img src="'+img+'.png" width=100 title="'+img+'">&nbsp;&nbsp;'

    TRAINING_GT = '''
    # Trained on ground truth data:
    #
    #   Training data:
    #   Patch size: (75,75)
    #   65000 correct splits
    #   65000 artificial split errors
    #   rotated 90,180,270 degrees after each epoch

    #   validation data: 10000 correct splits + 10000 split errors
    #   test data: 10000 correct splits + 10000 split errors
    '''

    TRAINING_RHOANA = '''
    # Trained on rhoana data (max overlap, difference in borders):
    #
    #   Training data:
    #   Patch size: (75,75)
    #   79828 correct splits
    #   79828 split errors
    #   rotated 90,180,270 degrees after each epoch
    '''

    if trained_gt:
      TRAINING = TRAINING_GT
    else:
      TRAINING = TRAINING_RHOANA


    #
    #
    # SPLIT TEST
    #
    #
    #
    print 'Performing split test..'
    for data in ['gt','rhoana']:

        global_vis, global_vi_diffs, global_surenesses, global_merge_pairs, global_best_indices, global_ugly_segmentations = Experiment.run_split_test(NETWORK,
                                                                                                                                  data=data,
                                                                                                                                  slices=SLICES,
                                                                                                                                  bbox=SPLIT_BBOX)


        mean_vi_diff = np.mean(global_vi_diffs)
        mean_vi = np.mean(global_vis)
        mean_sureness = np.mean(global_surenesses)

        best_vi_improvement_index = global_vi_diffs.index(max(global_vi_diffs))
        best_merge_pair_index = global_best_indices[best_vi_improvement_index]
        best_merge_pairs = global_merge_pairs[best_vi_improvement_index]
        best_ugly_segmentation = global_ugly_segmentations[best_vi_improvement_index]
        best_vi_diff = global_vi_diffs[best_vi_improvement_index]
        best_fixed_segmentation = Util.merge_steps(best_ugly_segmentation, best_merge_pairs, best=best_merge_pair_index, store=True)


        if data == 'gt':
            gt_mean_vi = mean_vi
            gt_mean_vi_diff = mean_vi_diff
            gt_mean_sureness = mean_sureness
            gt_best_vi_diff = best_vi_diff
            
        else:
            rhoana_mean_vi = mean_vi
            rhoana_mean_vi_diff = mean_vi_diff
            rhoana_mean_sureness = mean_sureness
            rhoana_best_vi_diff = best_vi_diff
                    
        Image.fromarray(Util.colorize(best_ugly_segmentation).astype(np.uint8)).save(REAL_OUTPUT_PATH+os.sep+'splits_'+data+'_best_ugly.png')
        Image.fromarray(Util.colorize(best_fixed_segmentation).astype(np.uint8)).save(REAL_OUTPUT_PATH+os.sep+'splits_'+data+'_best_fixed.png')
        
        pickle_stats = {
            'global_vis':global_vis,
            'global_vi_diffs':global_vi_diffs,
            'global_surenesses':global_surenesses,
            'global_merge_pairs':global_merge_pairs,
            'global_best_indices':global_best_indices,
            # 'global_eds':global_eds,
            # 'global_ed_diffs':global_ed_diffs,
            # 'global_surenesses_ed':global_surenesses_ed,            
        }
        
        with open(REAL_OUTPUT_PATH+os.sep+'splits_'+data+'_results.p', 'wb') as f:
            pickle.dump(pickle_stats, f)


    #
    #
    # MERGE TEST
    #
    #
    #
    print 'Performing merge test..'
    for data in ['gt','rhoana']:

      bins_sum = []
      global_vi_diffs = []
      global_bins = []
      for i in range(MERGE_ITERATIONS):
          vi_diffs, bins = Experiment.run_merge_test(NETWORK, bbox=MERGE_BBOX, N=MERGE_N, data=data, slices=SLICES)
          global_vi_diffs.append(vi_diffs)
          global_bins.append(bins)
          bins_sum.append(bins)

      total_bins = [0,0,0,0,0]
      if len(bins_sum[0]) > 0:
        total_bins_count = [0,0,0,0,0]
        for b in bins_sum:
            for i in range(5):
                total_bins[i] += b[i]
                total_bins_count[i] += 1

        for i in range(5):
            total_bins[i] /= total_bins_count[i]

      objects = ('Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5')
      y_pos = np.arange(len(objects))
      performance = total_bins

      fig = plt.figure()
       
      plt.bar(y_pos, performance, align='center', alpha=0.5)
      plt.xticks(y_pos, objects)
      plt.ylabel('Variation of Information difference (before-after)')
      # plt.title('')
       
      plt.savefig(REAL_OUTPUT_PATH+os.sep+'merge_'+data+'.png')

      pickle_stats = {
          'global_vi_diffs':global_vi_diffs,
          'global_bins':global_bins
      }

      if data == 'gt':
          gt_top1 = total_bins[0]
          gt_top2 = total_bins[1]
          gt_top3 = total_bins[2]
          gt_top4 = total_bins[3]
          gt_top5 = total_bins[4]
          
      else:
          rhoana_top1 = total_bins[0]
          rhoana_top2 = total_bins[1]
          rhoana_top3 = total_bins[2]
          rhoana_top4 = total_bins[3]
          rhoana_top5 = total_bins[4]
          
      
      with open(REAL_OUTPUT_PATH+os.sep+'merges_'+data+'_results.p', 'wb') as f:
          pickle.dump(pickle_stats, f)

    #
    # OUTPUT
    #
    #
    with open(OUTPUT_PATH+'template.html','r') as f:
        t = Template(f.read())
        t_out = t.substitute(CNN_NAME=CNN_NAME,
                            TRAINING=TRAINING,
                            CONFIGURATION=CONFIGURATION,
                            PATCHES=patches_html,
                            EPOCH=EPOCH,
                            TEST_LOSS=TEST_LOSS,
                            TEST_ACC=TEST_ACC,
                            SPLITS_GT_MEAN_VI=gt_mean_vi,
                            SPLITS_GT_MEAN_VI_DIFF=gt_mean_vi_diff,
                            SPLITS_GT_MEAN_SURENESS=gt_mean_sureness,
                            SPLITS_GT_BEST_VI_DIFF=gt_best_vi_diff,
                            SPLITS_RHOANA_MEAN_VI=rhoana_mean_vi,
                            SPLITS_RHOANA_MEAN_VI_DIFF=rhoana_mean_vi_diff,
                            SPLITS_RHOANA_MEAN_SURENESS=rhoana_mean_sureness,
                            SPLITS_RHOANA_BEST_VI_DIFF=rhoana_best_vi_diff,
                            MERGES_GT_TOP1=gt_top1,
                            MERGES_GT_TOP2=gt_top2,
                            MERGES_GT_TOP3=gt_top3,
                            MERGES_GT_TOP4=gt_top4,
                            MERGES_GT_TOP5=gt_top5,
                            MERGES_RHOANA_TOP1=rhoana_top1,
                            MERGES_RHOANA_TOP2=rhoana_top2,
                            MERGES_RHOANA_TOP3=rhoana_top3,
                            MERGES_RHOANA_TOP4=rhoana_top4,
                            MERGES_RHOANA_TOP5=rhoana_top5                            
                            )
        
    with open(REAL_OUTPUT_PATH+os.sep+'index.html', 'w') as f:
        f.write(t_out)
        
    template_vals = {
    'CNN_NAME':CNN_NAME,
    'TRAINING':TRAINING,
    'CONFIGURATION':CONFIGURATION,
    'PATCHES':patches_html,
    'EPOCH':EPOCH,
    'TEST_LOSS':TEST_LOSS,
    'TEST_ACC':TEST_ACC,
    'SPLITS_GT_MEAN_VI':gt_mean_vi,
    'SPLITS_GT_MEAN_VI_DIFF':gt_mean_vi_diff,
    'SPLITS_GT_MEAN_SURENESS':gt_mean_sureness,
    'SPLITS_GT_BEST_VI_DIFF':gt_best_vi_diff,
    'SPLITS_RHOANA_MEAN_VI':rhoana_mean_vi,
    'SPLITS_RHOANA_MEAN_VI_DIFF':rhoana_mean_vi_diff,
    'SPLITS_RHOANA_MEAN_SURENESS':rhoana_mean_sureness,
    'SPLITS_RHOANA_BEST_VI_DIFF':rhoana_best_vi_diff,
    'MERGES_GT_TOP1':gt_top1,
    'MERGES_GT_TOP2':gt_top2,
    'MERGES_GT_TOP3':gt_top3,
    'MERGES_GT_TOP4':gt_top4,
    'MERGES_GT_TOP5':gt_top5,
    'MERGES_RHOANA_TOP1':rhoana_top1,
    'MERGES_RHOANA_TOP2':rhoana_top2,
    'MERGES_RHOANA_TOP3':rhoana_top3,
    'MERGES_RHOANA_TOP4':rhoana_top4,
    'MERGES_RHOANA_TOP5':rhoana_top5
    }

    with open(REAL_OUTPUT_PATH+os.sep+'values.p', 'wb') as f:
        pickle.dump(template_vals, f)
        

    print 'All done.'

