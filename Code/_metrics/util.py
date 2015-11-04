import mahotas as mh
import numpy as np

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
    