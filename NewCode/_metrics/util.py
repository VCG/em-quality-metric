import glob
import h5py
import mahotas as mh
import numpy as np
import os
import tifffile as tif
from scipy import ndimage as nd
import partition_comparison
import matplotlib.pyplot as plt

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
    return mh.fullhistogram(array)

  @staticmethod
  def get_largest_label(array, ignore_zero=False):
    '''
    '''

    if ignore_zero:
      return np.argmax(Util.get_histogram(array)[1:])  
    return np.argmax(Util.get_histogram(array))

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
      section = tif.imread(f)
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
  def colorize(segmentation, colormap):
    '''
    '''
    pass

  @staticmethod
  def read_section(num):

    DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
    GOLD_PATH = os.path.join(DATA_PATH,'gold/')
    RHOANA_PATH = os.path.join(DATA_PATH,'rhoana/')
    IMAGE_PATH = os.path.join(DATA_PATH,'input/')
    PROB_PATH = os.path.join(DATA_PATH,'prob/')    
        
    gold = mh.imread(GOLD_PATH+'ac3_labels_00'+str(num)+'.tif')
    rhoana = mh.imread(RHOANA_PATH+os.sep+"z=000000"+str(num)+".tif")
    image = mh.imread(IMAGE_PATH+'ac3_input_00'+str(num)+'.tif')
    prob = mh.imread(PROB_PATH+'ac3_input_00'+str(num)+'_syn.tif')

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

