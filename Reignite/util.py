import glob
import h5py
import mahotas as mh
import numpy as np
import os
import random
import tifffile as tif
from scipy import ndimage as nd
import partition_comparison
import matplotlib.pyplot as plt
import skimage.measure
from scipy.spatial import distance


class Util(object):

  @staticmethod
  def threshold(array, value):
    '''
    '''
    output_array = np.zeros(array.shape)

    output_array[array == value] = 1

    return output_array

  @staticmethod
  def threshold_larger(array, value):
    '''
    '''
    output_array = np.zeros(array.shape)

    output_array[array >= value] = 1

    return output_array    

  @staticmethod
  def get_histogram(array):
    '''
    '''
    return mh.fullhistogram(array.astype(np.uint64))

  @staticmethod
  def get_largest_label(array, ignore_zero=False):
    '''
    '''

    hist = Util.get_histogram(array)

    if ignore_zero:
      hist[0] = 0


    return np.argmax(hist)

  @staticmethod
  def view_labels(array, labels, crop=True, large=True, return_it=False):

    if type(labels) != type(list()):
      labels = [labels]

    out = np.zeros(array.shape)

    for l in labels:

      l_arr = Util.threshold(array, l)
      out[l_arr == 1] = out.max()+1

    if crop:
      out = mh.croptobbox(out)

    if large:
      figsize = (10,10)
    else:
      figsize = (3,3)

    if return_it:
      return out

    fig = plt.figure(figsize=figsize)      

    plt.imshow(out)

    # return out

  @staticmethod
  def normalize_labels(array):
    '''
    '''
    return mh.labeled.relabel(array)

  @staticmethod
  def relabel(array):

    relabeled_array = np.array(array)
  
    relabeled_array = skimage.measure.label(array)
    relabeled_array[relabeled_array==0] = relabeled_array.max()
    
    return Util.normalize_labels(relabeled_array)[0]

  @staticmethod
  def read(path):
    '''
    Spits out a numpy volume in Z,Y,X format.
    '''
    files = glob.glob(path)

    volume_exists = False
    volume = None

    for i,f in enumerate(files):
      section = mh.imread(f)
      if not volume_exists:
        volume = np.zeros((len(files), section.shape[0], section.shape[1]))

        volume[0,:,:] = section
        volume_exists = True
      else:
        volume[i,:,:] = section

    print 'Loaded',len(files),'images.'

    return volume

  @staticmethod
  def fill(data, invalid=None):
      """
      Replace the value of invalid 'data' cells (indicated by 'invalid') 
      by the value of the nearest valid data cell

      Input:
          data:    numpy array of any dimension
          invalid: a binary array of same shape as 'data'. 
                   data value are replaced where invalid is True
                   If None (default), use: invalid  = np.isnan(data)

      Output: 
          Return a filled array. 
      """    
      if invalid is None: invalid = np.isnan(data)

      ind = nd.distance_transform_edt(invalid, 
                                      return_distances=False, 
                                      return_indices=True)
      return data[tuple(ind)]

  @staticmethod
  def load_colormap(f):
    '''
    '''
    hdf5_file = h5py.File(f, 'r')
    list_of_names = []
    hdf5_file.visit(list_of_names.append) 
    return hdf5_file[list_of_names[0]].value    

  @staticmethod
  def colorize(segmentation):
    '''
    '''
    cm_path = '/Volumes/DATA1/ac3x75/mojo/ids/colorMap.hdf5'
    if not os.path.exists(cm_path):
      cm_path = '/n/regal/pfister_lab/haehn/ac3x75/mojo/ids/colorMap.hdf5'
      cm_path = '/home/d/dojo_xp/data/colorMap.hdf5'


    cm = Util.load_colormap(cm_path)
    segmentation = cm[segmentation % len(cm)]
    return segmentation

  @staticmethod
  def read_section(num, keep_zeros=False, fill_zeros=False):

    DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
    if not os.path.isdir(DATA_PATH):
      DATA_PATH = '/n/regal/pfister_lab/haehn/ac3x75/'
    GOLD_PATH = os.path.join(DATA_PATH,'gold/')
    RHOANA_PATH = os.path.join(DATA_PATH,'rhoana/')
    IMAGE_PATH = os.path.join(DATA_PATH,'input/')
    PROB_PATH = os.path.join(DATA_PATH,'prob/')    
        
    gold = mh.imread(GOLD_PATH+'ac3_labels_00'+str(num)+'.tif')
    rhoana = mh.imread(RHOANA_PATH+os.sep+"z=000000"+str(num)+".tif")
    image = mh.imread(IMAGE_PATH+'ac3_input_00'+str(num)+'.tif')
    prob = mh.imread(PROB_PATH+'ac3_input_00'+str(num)+'_syn.tif')

    gold_original = np.array(gold)
    gold = Util.normalize_labels(skimage.measure.label(gold).astype(np.uint64))[0]
    gold[gold == 0] = gold.max()+1

    rhoana = Util.normalize_labels(skimage.measure.label(rhoana).astype(np.uint64))[0]

    # do we want to subtract the zeros?
    if keep_zeros:
      gold[gold_original == 0] = 0
      # rhoana[gold_original == 0] = 0

    if fill_zeros:
      gold[gold_original == 0] = 0
      gold_zeros = Util.threshold(gold, 0)
      gold = Util.fill(gold, gold_zeros.astype(np.bool))



    return image, prob, gold, rhoana

  @staticmethod
  def read_dojo_section(num, keep_zeros=False, fill_zeros=False, crop=True):

    DATA_PATH = '/Users/d/Projects/dojo_data_vis2014'
    if not os.path.isdir(DATA_PATH):
      DATA_PATH = '/n/regal/pfister_lab/haehn/dojo_data_vis2014/'
    GOLD_PATH = os.path.join(DATA_PATH,'groundtruth/')
    RHOANA_PATH = os.path.join(DATA_PATH,'labels_after_automatic_segmentation/')
    IMAGE_PATH = os.path.join(DATA_PATH,'images/')
    PROB_PATH = os.path.join(DATA_PATH,'prob/')    
        
    gold = tif.imread(GOLD_PATH+str(num)+'.tif')
    rhoana = tif.imread(RHOANA_PATH+os.sep+str(num)+".tif")
    image = tif.imread(IMAGE_PATH+str(num)+'.tif')
    prob = tif.imread(PROB_PATH+str(num)+'_syn.tif')

    gold_original = np.array(gold)
    gold = Util.normalize_labels(skimage.measure.label(gold).astype(np.uint64))[0]
    gold[gold == 0] = gold.max()+1

    rhoana = Util.normalize_labels(skimage.measure.label(rhoana).astype(np.uint64))[0]

    # do we want to subtract the zeros?
    if keep_zeros:
      gold[gold_original == 0] = 0
      # rhoana[gold_original == 0] = 0

    if fill_zeros:
      gold[gold_original == 0] = 0
      gold_zeros = Util.threshold(gold, 0)
      gold = Util.fill(gold, gold_zeros.astype(np.bool))

    if crop:
      bbox = mh.bbox(image)
      bbox_larger = [bbox[0]-37, bbox[1]+37, bbox[2]-37, bbox[3]+37]
      prob_new = prob
    else:
      bbox=bbox_larger = [0,1024,0,1024]      
      prob_new = np.zeros(image.shape, dtype=np.uint8)
      prob_new[bbox[0]:bbox[1], bbox[2]:bbox[3]] = prob[bbox[0]:bbox[1], bbox[2]:bbox[3]]

      


    return Util.crop_by_bbox(image, bbox_larger), Util.crop_by_bbox(prob_new, bbox_larger), Util.crop_by_bbox(gold, bbox_larger), Util.crop_by_bbox(rhoana, bbox_larger)


  @staticmethod
  def read_dojo_data():
    input_image = np.zeros((10,1024,1024))
    input_rhoana = np.zeros((10,1024,1024))
    input_gold = np.zeros((10,1024,1024))
    input_prob = np.zeros((10,1024,1024))
    path_prefix = '/Users/d/Projects/'
    path_prefix = '/home/d/dojo_xp/data/' # for beast only
    input_rhoana = tif.imread(path_prefix+'dojo_data_vis2014/labels_after_automatic_segmentation_multi.tif')
    input_gold = tif.imread(path_prefix+'dojo_data_vis2014/groundtruth_multi.tif')
    for i in range(10):
        input_prob[i] = tif.imread(path_prefix+'dojo_data_vis2014/prob/'+str(i)+'_syn.tif')
        input_image[i] = tif.imread(path_prefix+'dojo_data_vis2014/images/'+str(i)+'.tif')
        
    bbox = mh.bbox(input_image[0])
    bbox_larger = [bbox[0]-37, bbox[1]+37, bbox[2]-37, bbox[3]+37]

    prob_new = np.zeros(input_image.shape, dtype=np.uint8)
    
    input_image = input_image[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    input_rhoana = input_rhoana[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    input_gold = input_gold[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    # input_prob = input_prob[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    
    prob_new[:,bbox[0]:bbox[1], bbox[2]:bbox[3]] = input_prob[:,bbox[0]:bbox[1], bbox[2]:bbox[3]]
    prob_new = prob_new[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]



    for i in range(0,10):
      zeros_gold = Util.threshold(input_gold[i], 0)
      input_gold[i] = Util.normalize_labels(skimage.measure.label(input_gold[i]).astype(np.uint64))[0]
      # restore zeros
      input_gold[i][zeros_gold==1] = 0
      input_rhoana[i] = Util.normalize_labels(skimage.measure.label(input_rhoana[i]).astype(np.uint64))[0]

    return input_image.astype(np.uint8), prob_new.astype(np.uint8), input_gold.astype(np.uint32), input_rhoana.astype(np.uint32), bbox_larger


  @staticmethod
  def frame_image(image, shape=(75,75)):
    framed = np.array(image)
    framed[:shape[0]/2+1] = 0
    framed[-shape[0]/2+1:] = 0
    framed[:,0:shape[0]/2+1] = 0
    framed[:,-shape[0]/2+1:] = 0

    return framed

  @staticmethod
  def view(array,color=True,large=False,crop=False, text=None, filename=''):
    
    if large:
      figsize = (10,10)
    else:
      figsize = (3,3)

    fig = plt.figure(figsize=figsize)

    if crop:
      array = mh.croptobbox(array)

    if text:
      text = '\n\n\n'+str(text)
      fig.text(0,1,text)      

    


    if color:
      plt.imshow(Util.colorize(array), picker=True)
    else:
      plt.imshow(array, cmap='gray', picker=True)

    plt.savefig('/tmp/'+filename)

  @staticmethod
  def propagate_max_overlap(rhoana, gold):

      out = np.array(rhoana)
      
      rhoana_labels = Util.get_histogram(rhoana.astype(np.uint64))
      
      for l,k in enumerate(rhoana_labels):
          if l == 0 or k==0:
              # ignore 0 since rhoana does not have it
              continue
          values = gold[rhoana == l]
          largest_label = Util.get_largest_label(values.astype(np.uint64))
      
          out[rhoana == l] = largest_label # set the largest label from gold here
      
      return out

  @staticmethod 
  def erode_all(rhoana):

      out = np.zeros(rhoana.shape)

      rhoana_labels = Util.get_histogram(rhoana.astype(np.uint64))
      
      for l,k in enumerate(rhoana_labels):
          if l == 0 or k==0:
              # ignore 0 since rhoana does not have it
              continue

          isolated_label = Util.threshold(rhoana, l).astype(np.bool)
          for i in range(3):
            isolated_label = mh.erode(isolated_label)

          out[isolated_label == 1] = l


      return out

  @staticmethod
  def create_vi_plot(initial_segmentation, new_segmentation, vi_s, filepath=None):

    # create plot
    new_segmentation_target = Util.propagate_max_overlap(new_segmentation, initial_segmentation)
    before_vi = partition_comparison.variation_of_information(new_segmentation.ravel(), initial_segmentation.ravel())
    target_vi = partition_comparison.variation_of_information(new_segmentation_target.ravel(), initial_segmentation.ravel())

    bins = np.arange(0, len(vi_s))

    before_vi = [before_vi]*len(vi_s)
    target_vi = [target_vi]*len(vi_s)

    fig, ax = plt.subplots()

    ax.plot(bins, target_vi, 'k--', label='Target VI')
    ax.plot(bins, vi_s_, 'k', label='Variation of Information')
    ax.plot(bins, before_vi, 'k:', label='VI before')
    # ax.set_yscale('log')

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper center', shadow=True)
    ax.set_ylim([-0.5,1.5])

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    if filepath:
      plt.savefig(filepath)    

  @staticmethod
  def grab_neighbors(array, label):

      thresholded_array = Util.threshold(array, label)
      thresholded_array_dilated = mh.dilate(thresholded_array.astype(np.uint64))

      copy = np.array(array)
      copy[thresholded_array_dilated != thresholded_array_dilated.max()] = 0
      copy[thresholded_array == 1] = 0

      copy_hist = Util.get_histogram(copy.astype(np.uint64))

      copy_hist[0] = 0 # ignore zeros
      # copy_hist[label] = 0 # ignore ourselves
      return np.where(copy_hist>0)[0]

  @staticmethod
  def vi(array1, array2):
    '''
    '''
    return partition_comparison.variation_of_information(array1.ravel(), array2.ravel())

  @staticmethod
  def merge_steps(array, merge_pairs, best=-1, snapshot_interval=50, store=False):
    '''
    '''
    state_array = np.array(array)

    for i,m in enumerate(merge_pairs):
        
        l,n = m
        
        state_array[state_array == n] = l
        
        
        if best != -1 and i == best:
          if store:
            return state_array
          Util.view(state_array, large=True)

        elif best == -1 and i % snapshot_interval == 0:
          Util.view(state_array, large=True)


  @staticmethod
  def dice(array, window=(300,300)):
    '''
    Return a list of diced windows of an array.
    '''
    windows = []

    for y in range(0,array.shape[0],window[0])[:-1]:
      for x in range(0,array.shape[1],window[1])[:-1]:

        bbox = [y,y+window[0],x,x+window[1]]
        subarray = array[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        windows.append(subarray)

    return windows


  @staticmethod
  def stats(results):
    print '  N: ', len(results)
    print '  Min: ', np.min(results)
    print '  Max: ', np.max(results)
    print '  Mean: ', np.mean(results)
    print '  Median: ', np.median(results)
    print '  Std: ', np.std(results)
    print '  Var: ', np.var(results)


  @staticmethod
  def crop_by_bbox(array, bbox):

    return array[bbox[0]:bbox[1], bbox[2]:bbox[3]]

  @staticmethod
  def gradient(array, sigma=2.5):
    '''
    '''

    grad = mh.gaussian_filter(array, sigma)

    grad_x = np.gradient(grad)[0]
    grad_y = np.gradient(grad)[1]
    grad = np.sqrt(np.add(grad_x*grad_x, grad_y*grad_y))

    grad -= grad.min()
    grad /= (grad.max() - grad.min())
    grad *= 255

    return grad

  @staticmethod
  def invert(array, smooth=False, sigma=2.5):
    
    grad = mh.gaussian_filter(array, sigma)

    return (255-grad)

  @staticmethod
  def random_watershed(array, speed_image, border_seeds=False, erode=False):
    '''
    '''
    copy_array = np.array(array, dtype=np.bool)

    if erode:
      
      for i in range(10):
        copy_array = mh.erode(copy_array)


    seed_array = np.array(copy_array)
    if border_seeds:
      seed_array = mh.labeled.border(copy_array, 1, 0, Bc=mh.disk(7))

    coords = zip(*np.where(seed_array==1))

    if len(coords) == 0:
      # print 'err'
      return np.zeros(array.shape)

    seed1_ = None
    seed2_ = None
    max_distance = -np.inf

    for i in range(10):
      seed1 = random.choice(coords)
      seed2 = random.choice(coords)
      d = distance.euclidean(seed1, seed2)
      if max_distance < d:
        max_distance = d
        seed1_ = seed1
        seed2_ = seed2

    seeds = np.zeros(array.shape, dtype=np.uint8)
    seeds[seed1_[0], seed1_[1]] = 1
    seeds[seed2_[0], seed2_[1]] = 2



    for i in range(8):
      seeds = mh.dilate(seeds)

    # Util.view(seeds,large=True)      
    # print speed_image.shape, seeds.shape
    ws = mh.cwatershed(speed_image, seeds)
    ws[array == 0] = 0

    return ws

  @staticmethod
  def dark_watershed(image, seed_image, threshold=50., dilate=True):
    '''
    '''

    coords = zip(*np.where(seed_image==threshold))
    print 'seeds:', len(coords)
    seeds = np.zeros(seed_image.shape, dtype=np.uint64)
    for c in coords:
      seeds[c[0], c[1]] = seeds.max()+1

    if dilate:
      for i in range(8):
        seeds = mh.dilate(seeds)

    ws = mh.cwatershed(image, seeds)

    return seeds,ws



  @staticmethod
  def show_borders(image, segmentation):

    borders = mh.labeled.borders(segmentation)

    b = np.zeros((1024,1024,3), dtype=np.uint8)
    b[:,:,0] = image[:]
    b[:,:,1] = image[:]
    b[:,:,2] = image[:]

    b[borders==1] = (255,0,0)
    Util.view(b, color=False, large=True)

  @staticmethod
  def show_overlay(image, segmentation, borders=np.zeros((1,1)), labels=np.zeros((1,1)),mask=None):

    b = np.zeros((image.shape[0],image.shape[1],4), dtype=np.uint8)
    c = np.zeros((image.shape[0],image.shape[1],4), dtype=np.uint8)
    b[:,:,0] = image[:]
    b[:,:,1] = image[:]
    b[:,:,2] = image[:]
    b[:,:,3] = 255
    # from PIL import Image
    # def alpha_composite(src, dst):
    #     '''
    #     Return the alpha composite of src and dst.

    #     Parameters:
    #     src -- PIL RGBA Image object
    #     dst -- PIL RGBA Image object

    #     The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    #     '''
    #     # http://stackoverflow.com/a/3375291/190597
    #     # http://stackoverflow.com/a/9166671/190597
    #     src = np.asarray(src)
    #     dst = np.asarray(dst)
    #     out = np.empty(src.shape, dtype = 'float')
    #     alpha = np.index_exp[:, :, 3:]
    #     rgb = np.index_exp[:, :, :3]
    #     src_a = src[alpha]/255.0
    #     dst_a = dst[alpha]/255.0
    #     out[alpha] = src_a+dst_a*(1-src_a)
    #     old_setting = np.seterr(invalid = 'ignore')
    #     out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
    #     np.seterr(**old_setting)    
    #     out[alpha] *= 255
    #     np.clip(out,0,255)
    #     # astype('uint8') maps np.nan (and np.inf) to 0
    #     out = out.astype('uint8')
    #     out = Image.fromarray(out, 'RGBA')
    #     return out

    if not labels.shape[0]>1:
      # c[segmentation==1] = (00,0,200,130)
      # c[segmentation==2] = (0,150,00,130)
      # c[mask!=0] = (0,0,200,130)
      c[segmentation==1] = (0,150,0,130)
      c[segmentation==2] = (200,0,000,130)
      c[segmentation==3] = (100,100,00,130)
      c[segmentation==4] = (0,0,200,130)      
    if borders.shape[0]>1:
      borders[mh.erode(mh.erode(mh.erode(segmentation)))==0] = 0
      c[borders==borders.max()] = (0,255,0,255)
      c[borders==borders.max()-1] = (255,0,0,255)
    elif labels.shape[0]>1:
      c[mask!=0] = (0,0,200,130)
      c[labels==1] = (0,150,0,130)
      c[labels==2] = (200,0,000,130)
      c[labels==3] = (100,100,00,130)
      c[labels==4] = (0,0,200,130)
    return b,c

    # return alpha_composite(Image.fromarray(b, 'RGBA'),Image.fromarray(c, 'RGBA'))

  @staticmethod
  def ed(gt_stack, seg_stack):
      min_2d_seg_size = 500
      min_3d_seg_size = 2000    
      
      gt_ids = np.unique(gt_stack.ravel())
      
      seg_ids = np.unique(seg_stack.ravel())

      # count 2d split operations required
      split_count_2d = 0
      for seg_id in seg_ids:
          if seg_id == 0:
              continue
          if seg_stack.ndim == 2:

                gt_counts = np.bincount(gt_stack[:,:][seg_stack[:,:]==seg_id])
                if len(gt_counts) == 0:
                    continue
                gt_counts[0] = 0
                gt_counts[gt_counts < min_2d_seg_size] = 0
                gt_objects = len(np.nonzero(gt_counts)[0])
                if gt_objects > 1:
                    split_count_2d += gt_objects - 1

          else:
            for zi in range(seg_stack.shape[0]):
                gt_counts = np.bincount(gt_stack[zi,:,:][seg_stack[zi,:,:]==seg_id])
                if len(gt_counts) == 0:
                    continue
                gt_counts[0] = 0
                gt_counts[gt_counts < min_2d_seg_size] = 0
                gt_objects = len(np.nonzero(gt_counts)[0])
                if gt_objects > 1:
                    split_count_2d += gt_objects - 1

      # count 3d split operations required
      split_count_3d = 0
      for seg_id in seg_ids:
          if seg_id == 0:
              continue
          gt_counts = np.bincount(gt_stack[seg_stack==seg_id])
          if len(gt_counts) == 0:
              continue
          gt_counts[0] = 0
          gt_counts[gt_counts < min_3d_seg_size] = 0
          gt_objects = len(np.nonzero(gt_counts)[0])
          if gt_objects > 1:
              split_count_3d += gt_objects - 1

      # count 3d merge operations required
      merge_count = 0
      for gt_id in gt_ids:
          if gt_id == 0:
              continue
          seg_counts = np.bincount(seg_stack[gt_stack==gt_id])
          if len(seg_counts) == 0:
              continue
          seg_counts[0] = 0
          seg_counts[seg_counts < min_3d_seg_size] = 0
          seg_objects = len(np.nonzero(seg_counts)[0])
          if seg_objects > 1:
              merge_count += seg_objects - 1

      # print "{0} 2D Split or {1} 3D Split and {2} 3D Merge operations required.".format(split_count_2d, split_count_3d, merge_count)

      # return (split_count_2d, split_count_3d, merge_count)    
      return split_count_2d+merge_count





