import mahotas as mh
import numpy as np

from patch import Patch
from util import Util

class Uglify(object):

  @staticmethod
  def split(image, prob, segmentation, n=1, min_pixels=100):
    '''
    '''
    ugly_segmentation = np.array(segmentation)

    hist = Util.get_histogram(ugly_segmentation.astype(np.uint64))
    labels = len(hist)
    

    for l in range(labels):

      if l == 0:
        continue

      for s in range(n):

        binary_mask = Util.threshold(ugly_segmentation, l)

        splitted_label, border = Patch.split_label(image, binary_mask)

        # check if splitted_label is large enough
        if len(splitted_label[splitted_label == 1]) < min_pixels:
          continue

        ugly_segmentation[splitted_label == 1] = ugly_segmentation.max()+1

    return ugly_segmentation





  @staticmethod
  def merge(image, prob, segmentation):
    '''
    '''
    pass

  @staticmethod
  def merge_label(image, prob, segmentation, label1, label2, crop=True, patch_size=(100,100)):
    '''
    '''
    copy_segmentation = np.array(segmentation)

    border = mh.labeled.border(segmentation, label1, label2)

    copy_segmentation[copy_segmentation == label2] = label1

    binary = Util.threshold(copy_segmentation, label1)

    if crop:
      bbox = mh.bbox(Util.threshold(copy_segmentation, label1))
      bbox = [bbox[0]-patch_size[0], bbox[1]+patch_size[0], bbox[2]-patch_size[1], bbox[3]+patch_size[1]]

      cropped_image = Util.crop_by_bbox(image, bbox)
      cropped_binary = Util.crop_by_bbox(binary, bbox)
      cropped_prob = Util.crop_by_bbox(prob, bbox)
      cropped_segmentation = Util.crop_by_bbox(copy_segmentation, bbox)

      return cropped_image, cropped_prob, cropped_segmentation, cropped_binary, bbox, Util.crop_by_bbox(border, bbox)

    else:

      return image, prob, copy_segmentation, binary, [0, segmentation.shape[0], 0, segmentation.shape[1]]



