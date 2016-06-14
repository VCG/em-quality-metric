import glob
import h5py
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.measure



class Util(object):

  @staticmethod
  def read_section(path, z):
    '''
    '''
    image = sorted(glob.glob(os.path.join(path, 'image', '*'+str(z)+'.png')))
    mask = sorted(glob.glob(os.path.join(path, 'mask', '*'+str(z)+'.png')))   
    gold = sorted(glob.glob(os.path.join(path, 'gold', '*'+str(z)+'.png')))
    rhoana = sorted(glob.glob(os.path.join(path, 'rhoana', '*'+str(z)+'.png')))
    prob = sorted(glob.glob(os.path.join(path, 'prob', '*'+str(z)+'_syn.tif')))

    print 'Loading', os.path.basename(image[0])

    image = mh.imread(image[0])
    mask = mh.imread(mask[0]).astype(np.bool)
    gold = mh.imread(gold[0])
    rhoana = mh.imread(rhoana[0])
    prob = mh.imread(prob[0])

    #convert ids from rgb to single channel
    rhoana_single = np.zeros((rhoana.shape[0], rhoana.shape[1]), dtype=np.uint64)
    rhoana_single[:, :] = rhoana[:,:,0]*256*256 + rhoana[:,:,1]*256 + rhoana[:,:,2]
    gold_single = np.zeros((gold.shape[0], gold.shape[1]), dtype=np.uint64)
    gold_single[:, :] = gold[:,:,0]*256*256 + gold[:,:,1]*256 + gold[:,:,2]

    # relabel the segmentations
    gold_single = Util.relabel(gold_single)
    rhoana_single = Util.relabel(rhoana_single)


    #mask the rhoana output
    rhoana_single[mask==0] = 0


    return image, prob, mask, gold_single, rhoana_single

      
  @staticmethod
  def get_histogram(array):
    '''
    '''
    return mh.fullhistogram(array.astype(np.uint64))

  @staticmethod
  def get_largest_label(array, ignore_zero=False):
    '''
    '''

    hist = Util.get_histogram(array)

    if ignore_zero:
      hist[0] = 0


    return np.argmax(hist)

  @staticmethod
  def normalize_labels(array):
    '''
    '''
    return mh.labeled.relabel(array)

  @staticmethod
  def relabel(array):

    relabeled_array = np.array(array)
  
    relabeled_array = skimage.measure.label(array).astype(np.uint64)
    # relabeled_array[relabeled_array==0] = relabeled_array.max()
    
    return Util.normalize_labels(relabeled_array)[0]

  @staticmethod
  def load_colormap(f):
    '''
    '''
    hdf5_file = h5py.File(f, 'r')
    list_of_names = []
    hdf5_file.visit(list_of_names.append) 
    return hdf5_file[list_of_names[0]].value    

  @staticmethod
  def colorize(segmentation):
    '''
    '''
    cm_path = '/home/d/data/colorMap.hdf5'

    cm = Util.load_colormap(cm_path)
    segmentation = cm[segmentation % len(cm)]
    return segmentation

  @staticmethod
  def view(array,color=True,large=False,crop=False, text=None):
    
    if large:
      figsize = (10,10)
    else:
      figsize = (3,3)

    fig = plt.figure(figsize=figsize)

    if crop:
      array = mh.croptobbox(array)

    if text:
      text = '\n\n\n'+str(text)
      fig.text(0,1,text)      


    if color:
      plt.imshow(Util.colorize(array), picker=True)
    else:
      plt.imshow(array, cmap='gray', picker=True)

  @staticmethod
  def view_labels(array, labels, crop=True, large=True, return_it=False):

    if type(labels) != type(list()):
      labels = [labels]

    out = np.zeros(array.shape)

    for l in labels:

      l_arr = Util.threshold(array, l)
      out[l_arr == 1] = out.max()+1

    if crop:
      out = mh.croptobbox(out)

    if large:
      figsize = (10,10)
    else:
      figsize = (3,3)

    if return_it:
      return out

    fig = plt.figure(figsize=figsize)      

    plt.imshow(out)


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
  def grab_neighbors(array, label):

      thresholded_array = Util.threshold(array, label)
      thresholded_array_dilated = mh.dilate(thresholded_array.astype(np.uint64))

      copy = np.array(array)
      copy[thresholded_array_dilated != thresholded_array_dilated.max()] = 0
      copy[thresholded_array == 1] = 0

      copy_hist = Util.get_histogram(copy.astype(np.uint64))

      copy_hist[0] = 0 # ignore zeros
      # copy_hist[label] = 0 # ignore ourselves
      return np.where(copy_hist>0)[0]


  @staticmethod
  def threshold(array, value):
    '''
    '''
    output_array = np.zeros(array.shape)

    output_array[array == value] = 1

    return output_array
    