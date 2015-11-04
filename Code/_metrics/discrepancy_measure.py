from metric import Metric
import numpy as np

class DiscrepancyMeasure(Metric):
  '''
  This implements the Discrepancy Measure D_WR introduced by Weszka et Rosenfeld in 1978.

  @ARTICLE{4310038, 
  author={Weszka, Joan S. and Rosenfeld, Azriel}, 
  journal={Systems, Man and Cybernetics, IEEE Transactions on}, 
  title={Threshold Evaluation Techniques}, 
  year={1978}, 
  volume={8}, 
  number={8}, 
  pages={622-629}, 
  keywords={Biological cells;Blood;Clouds;Computational efficiency;Computer science;Cost function;Histograms;Image segmentation;Infrared imaging;Layout}, 
  doi={10.1109/TSMC.1978.4310038}, 
  ISSN={0018-9472}, 
  month={Aug},}
  '''

  @staticmethod
  def apply2D(label_array, image_array):
    '''
    '''
    # mask the image_array using the label_array
    masked_array = np.array(image_array)
    masked_array[label_array == 0] = 0

    return np.sum(image_array[:,:] - masked_array[:,:])
