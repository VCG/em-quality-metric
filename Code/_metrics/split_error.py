import cv2
import cPickle as pickle
import numpy as np
import os
import scipy.misc
import skimage.measure
from skimage import img_as_ubyte


from error import Error
from util import Util

class SplitError(Error):
    '''
    '''
    def __init__(self):
        super(SplitError, self).__init__()
        
    def rotate(self, degrees):
        '''
        '''
        s = SplitError()
        s._meta = {}
        s._meta['merge'] = scipy.misc.imrotate(self._meta['merge'], degrees, interp='nearest')
        s._meta['image'] = scipy.misc.imrotate(self._meta['image'], degrees, interp='nearest')
        s._meta['prob'] = scipy.misc.imrotate(self._meta['prob'], degrees, interp='nearest')
        s._meta['label1'] = scipy.misc.imrotate(self._meta['label1'], degrees, interp='nearest')
        s._meta['label2'] = scipy.misc.imrotate(self._meta['label2'], degrees, interp='nearest')
        s._meta['overlap'] = scipy.misc.imrotate(self._meta['overlap'], degrees, interp='nearest')

        if self._has_thumb:
            s._has_thumb = True
            s._thumb = scipy.misc.imrotate(self._thumb, degrees, interp='nearest')

        return s


    def fliplr(self):
        '''
        '''
        s = SplitError()
        s._meta = {}
        s._meta['merge'] = np.fliplr(self._meta['merge'])#cv2.flip(self._meta['merge'], how)
        s._meta['image'] = np.fliplr(self._meta['image'])#cv2.flip(self._meta['image'], how)
        s._meta['prob'] = np.fliplr(self._meta['prob'])#cv2.flip(self._meta['prob'], how)
        s._meta['label1'] = np.fliplr(self._meta['label1'])#cv2.flip(self._meta['label1'], how)
        s._meta['label2'] = np.fliplr(self._meta['label2'])#cv2.flip(self._meta['label2'], how)
        s._meta['overlap'] = np.fliplr(self._meta['overlap'])#cv2.flip(self._meta['overlap'], how)

        if self._has_thumb:
            s._has_thumb = True
            s._thumb = np.fliplr(self._thumb)

        return s

    def flipud(self):
        '''
        '''
        s = SplitError()
        s._meta = {}
        s._meta['merge'] = np.flipud(self._meta['merge'])#cv2.flip(self._meta['merge'], how)
        s._meta['image'] = np.flipud(self._meta['image'])#cv2.flip(self._meta['image'], how)
        s._meta['prob'] = np.flipud(self._meta['prob'])#cv2.flip(self._meta['prob'], how)
        s._meta['label1'] = np.flipud(self._meta['label1'])#cv2.flip(self._meta['label1'], how)
        s._meta['label2'] = np.flipud(self._meta['label2'])#cv2.flip(self._meta['label2'], how)
        s._meta['overlap'] = np.flipud(self._meta['overlap'])#cv2.flip(self._meta['overlap'], how)

        if self._has_thumb:
            s._has_thumb = True
            s._thumb = np.flipud(self._thumb)

        return s    

    @staticmethod
    def load(uuid):
        folder = os.path.join(TRAINING_PATH,'split')
        m = SplitError()
        m._meta = pickle.load(open(os.path.join(folder, uuid+'.p'), 'rb'))
                              #np.load(os.path.join(folder, uuid+'.npz'))
        m._thumb = cv2.imread(os.path.join(folder, uuid+'.tif'))
        
        return m        
        
    @staticmethod
    def create(image, prob, segmentation, label, thumb=True):
        '''
        '''
        m = SplitError()
        ws = m.split(image, segmentation, label)
        m._meta = m.analyze_border(image, prob, ws, label, ws.max()-1, ws.max())
        if thumb:
            m._has_thumb = True
            m._thumb = m.create_thumb(image, m._meta)
        
        if not m._meta:
            return None        
        
        return m
    
    @staticmethod
    def store_many(training_path, image, segmentation, label, n=10):
        '''
        '''
        for i in range(n):
            s = SplitError.create(image, segmentation, label)
            while not s:
                s = SplitError.create(image, segmentation, label)
            s.store(os.path.join(training_path,'split'))
        
    @staticmethod
    def generate(image, prob, label, n=10, thumb=False, rotate=True, flip=True):
        '''
        '''

        # run through all slices
        for z in range(image.shape[0]):

            # fill segmentation
            label_zeros = Util.threshold(label[z], 0)
            label_filled = Util.fill(label[z], label_zeros.astype(np.bool))
            label_filled_relabeled = skimage.measure.label(label_filled).astype(np.uint64)

            print 'Working on z', z

            labels = range(len(Util.get_histogram(label_filled_relabeled)))[1:] ### remove!!
            for l in labels:

                for i in range(n):

                    upper_limit = 3

                    s = SplitError.create(image[z], prob[z], label_filled_relabeled, l, thumb)
                    while not s:

                        if upper_limit == 0:
                          #print "Upper limit reached."
                          break 

                        s = SplitError.create(image[z], prob[z], label_filled_relabeled, l, thumb)
                        upper_limit -= 1

                    if not s:
                      continue

                    yield s

                    if flip:
                        yield s.fliplr()
                        yield s.flipud()

                    if rotate:
                        yield s.rotate(90)
                        yield s.rotate(180)
                        m270 = s.rotate(270)
                        yield m270
                        if flip:
                            yield m270.fliplr()
                            yield m270.flipud()



