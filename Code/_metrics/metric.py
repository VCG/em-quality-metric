import numpy as np

class Metric(object):
  '''
  The super class for all (segmentation) evaluation metrics.
  '''

  def __init__(self):
    '''
    '''
    pass

  @staticmethod
  def apply2D(label_array, image_array=np.zeros(0)):
    '''
    Apply the metric to a label_array with shape Y,X.
    '''
    return 0

  @staticmethod
  def apply(label_array, image_array=np.zeros(0)):
    '''
    Apply the metric to a label_array with shape Y,X,Z.
    '''

    twoD_sum = 0


    if label_array.ndim == 2:
      return cls.apply2D(label_array[:,:], image_array[:,:])

    for z in label_array.shape[2]:
      twoD_sum += cls.apply2D(label_array[:,:,z])
    
    return twoD_sum

