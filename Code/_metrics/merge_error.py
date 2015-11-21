import cv2
import cPickle as pickle
import mahotas as mh
import numpy as np
import os

from error import Error
from util import Util

class MergeError(Error):
    '''
    '''
    def __init__(self):
        super(MergeError, self).__init__()
        
    @staticmethod
    def grab_neighbors(array, label):

        thresholded_array = Util.threshold(array, label)
        thresholded_array_dilated = mh.dilate(thresholded_array.astype(np.uint64))

        all_neighbors = np.unique(array[thresholded_array_dilated == thresholded_array_dilated.max()].astype(np.uint64))
        all_neighbors = np.delete(all_neighbors, np.where(all_neighbors == label))

    #     neighbors = {}

    #     for n in all_neighbors:
    #         border = mh.labeled.border(array, label, n)
    #         border_xy = np.where(border==True)
    #         neighbors[str(n)] = border_xy

        return all_neighbors        

    @staticmethod
    def load(uuid):
        folder = os.path.join(TRAINING_PATH,'merge')        
        m = MergeError()
        m._meta = pickle.load(open(os.path.join(folder, uuid+'.p'), 'rb'))
                              #np.load(os.path.join(folder, uuid+'.npz'))
        m._thumb = cv2.imread(os.path.join(folder, uuid+'.tif'))
        
        return m         
    
    @staticmethod
    def create(image, segmentation, label1, label2):
        '''
        '''
        m = MergeError()
        m._meta = m.analyze_border(segmentation, label1, label1, label2)
        m._thumb = m.create_thumb(image, m._meta)
        
        if not m._meta:
            return None
        
        return m
        
    @staticmethod
    def store_many(training_path, image, segmentation, label):
        '''
        '''
        neighbors = MergeError.grab_neighbors(segmentation, label)
        for n in neighbors:
            m = MergeError.create(image, segmentation, label, n)
            if m:
                m.store(os.path.join(training_path,'merge'))
            else:
                print 'could not do', label, n
            #while not m:
            #    m = MergeError.create(image, segmentation, label, n)
            
        
