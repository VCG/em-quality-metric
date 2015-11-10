from metric import Metric
from util import Util
import numpy as np

class SizeDistributionMeasure(Metric):
  '''
  This calculates statistical information based on the size of elements.
  '''

  @classmethod
  def apply(cls, label_array, normalize=True):
    '''
    Apply the metric to a label_array with shape Y,X,Z.
    '''

    #print 'Calulating for',label_array.shape[0],'slices.'

    # normalize the input array and get the histogram
    if normalize:
      label_array = Util.normalize_labels(label_array)[0]
      
    # we do ignore label 0
    hist = Util.get_histogram(label_array.astype(np.uint64))[1:]

    output = {}
    output['Count'] = len(hist)
    output['Min'] = np.min(hist)
    output['Max'] = np.max(hist)
    output['Median'] = np.median(hist)
    output['Average'] = np.average(hist)
    output['Mean'] = np.mean(hist)
    output['SD'] = np.std(hist)
    output['Variance'] = np.var(hist)

    return output
