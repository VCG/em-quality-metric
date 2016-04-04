import mahotas as mh
import numpy as np
import scipy.misc
import skimage.measure

from util import Util
from patch import Patch

class Generator(object):

  @staticmethod
  def generate_split_error(image, prob, segmentation, n=3):
    
    hist = Util.get_histogram(segmentation.astype(np.uint64))
    labels = range(1,len(hist)) # do not include zeros

    np.random.shuffle(labels)

    for l in labels:

        binary_mask = Util.threshold(segmentation, l)
        labeled_parts = skimage.measure.label(binary_mask)
        labeled_parts += 1 # avoid the 0
        labeled_parts[binary_mask == 0] = 0
        labeled_parts, no_labeled_parts = mh.labeled.relabel(labeled_parts)


        for i in range(n):

          for part in range(1,no_labeled_parts+1):

              binary_part = Util.threshold(labeled_parts, part)
              split_binary_mask, split_isolated_border = Patch.split_label(image, binary_part)

              if split_binary_mask.max() == 0:
                # the strange err (label too small or sth)
                print 'Caught empty..'
                continue

              patches = Patch.analyze_border(image, prob, split_binary_mask, binary_mask.invert(), split_isolated_border)

              for s in patches:

                  yield s
                  yield Patch.fliplr(s)
                  yield Patch.flipud(s)
                  yield Patch.rotate(s, 90)
                  yield Patch.rotate(s, 180)
                  yield Patch.rotate(s, 270)  



  @staticmethod
  def generate_correct(image, prob, segmentation):

    hist = Util.get_histogram(segmentation.astype(np.uint64))
    labels = range(1,len(hist)) # do not include zeros

    np.random.shuffle(labels)

    for l in labels:

        binary_mask = Util.threshold(segmentation, l)
        borders = mh.labeled.borders(binary_mask,Bc=mh.disk(2))
        labeled_borders = skimage.measure.label(borders)
        labeled_borders[borders==0] = 0
        relabeled_borders, no_relabeled_borders = mh.labeled.relabel(labeled_borders.astype(np.uint16))

        for border in range(1,no_relabeled_borders+1):
        
            isolated_border = Util.threshold(relabeled_borders, border)

            patches = Patch.analyze_border(image, prob, binary_mask, binary_mask.invert(), isolated_border)

            for s in patches:

                yield s
                yield Patch.fliplr(s)
                yield Patch.flipud(s)
                yield Patch.rotate(s, 90)
                yield Patch.rotate(s, 180)
                yield Patch.rotate(s, 270)  
