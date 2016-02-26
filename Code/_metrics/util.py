import glob
import h5py
import mahotas as mh
import numpy as np
import os
import tifffile as tif
from scipy import ndimage as nd


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

