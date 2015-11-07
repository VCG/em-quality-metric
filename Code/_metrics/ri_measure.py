from metric import Metric
from util import Util
import numpy as np
import partition_comparison

class RIMeasure(Metric):
  '''
  This calculates rand index.
  '''

  @classmethod
  def apply(cls, label_array):
    '''
    Apply the metric to a label_array with shape Y,X,Z.
    '''

    vi_sum = 0.
    done = 0.

    print 'Calulating for',label_array.shape[0],'slices.'

    for z in range(label_array.shape[0]-1):
      vi_sum += partition_comparison.rand_index(label_array[z,:,:].astype(np.uint64).ravel(),
                                                label_array[z+1,:,:].astype(np.uint64).ravel())

      done += 1
      percentage =  int((done / (label_array.shape[0]-1))*100)
      if (percentage % 10 == 0):
        print percentage, '% done'
    
    vi_sum /= label_array.shape[0]

    return vi_sum
