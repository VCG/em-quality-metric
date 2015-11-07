import glob
import mahotas as mh
import numpy as np
import tifffile as tif

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
  def get_largest_label(array):
    '''
    '''
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
