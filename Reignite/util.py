import glob
import h5py
import mahotas as mh
import numpy as np
import os
import random
# import tifffile as tif
from scipy import ndimage as nd
import partition_comparison
import matplotlib.pyplot as plt
import skimage.measure


class Util(object):

  @staticmethod
  def threshold(array, value):
    '''
    '''
    output_array = np.zeros(array.shape)

    output_array[array == value] = 1

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
  def view_labels(array, labels, crop=True, large=True):

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

    fig = plt.figure(figsize=figsize)      

    plt.imshow(out)

    # return out

  @staticmethod
  def normalize_labels(array):
    '''
    '''
    return mh.labeled.relabel(array)

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
    cm = Util.load_colormap('/Volumes/DATA1/ac3x75/mojo/ids/colorMap.hdf5')
    segmentation = cm[segmentation % len(cm)]
    return segmentation

  @staticmethod
  def read_section(num, keep_zeros=False):

    DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
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


    return image, prob, gold, rhoana

  @staticmethod
  def frame_image(image, shape=(75,75)):
    framed = np.array(image)
    framed[:shape[0]/2+1] = 0
    framed[-shape[0]/2+1:] = 0
    framed[:,0:shape[0]/2+1] = 0
    framed[:,-shape[0]/2+1:] = 0

    return framed

  @staticmethod
  def view(array,color=True,large=False,crop=False, text=None):
    
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
  def merge_steps(array, merge_pairs, best=-1, snapshot_interval=50):
    '''
    '''
    state_array = np.array(array)

    for i,m in enumerate(merge_pairs):
        
        l,n = m
        
        state_array[state_array == n] = l
        
        
        if best != -1 and i == best:
          Util.view(state_array, large=True)

        elif best == -1 and i % snapshot_interval == 0:
          Util.view(state_array, large=False)


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
  def invert(array):
    
    return (255-array)

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
      seed_array = mh.labeled.border(copy_array, 1, 0)

    coords = zip(*np.where(seed_array==1))


    seed1 = random.choice(coords)
    seed2 = random.choice(coords)

    seeds = np.zeros(array.shape, dtype=np.uint8)
    seeds[seed1[0], seed1[1]] = 1
    seeds[seed2[0], seed2[1]] = 2

    for i in range(8):
      seeds = mh.dilate(seeds)

    ws = mh.cwatershed(speed_image, seeds)
    ws[array == 0] = 0

    return ws

















