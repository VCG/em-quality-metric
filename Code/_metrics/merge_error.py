import cv2
import cPickle as pickle
import mahotas as mh
import numpy as np
import os
import cv2
import cPickle as pickle
import numpy as np
import os
import scipy.misc
import skimage.measure

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


    def rotate(self, degrees):
        '''
        '''
        s = MergeError()
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
        s = MergeError()
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
        s = MergeError()
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
        folder = os.path.join(TRAINING_PATH,'merge')        
        m = MergeError()
        m._meta = pickle.load(open(os.path.join(folder, uuid+'.p'), 'rb'))
                              #np.load(os.path.join(folder, uuid+'.npz'))
        m._thumb = cv2.imread(os.path.join(folder, uuid+'.tif'))
        
        return m         
    
    @staticmethod
    def create(image, prob, segmentation, label1, label2, thumb=True):
        '''
        '''
        m = MergeError()
        # return segmentation, label1, label2
        m._meta = m.analyze_border(image, prob, segmentation, label1, label1, label2)
        
        if thumb:
            m._has_thumb = True
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
            
        
    # @staticmethod
    # def generate(image, prob, label, thumb=False, rotate=True, flip=True):
    #     '''
    #     '''

    #     # run through all slices
    #     for z in range(image.shape[0]):

    #         # fill segmentation
    #         label_zeros = Util.threshold(label[z], 0)
    #         label_filled = Util.fill(label[z], label_zeros.astype(np.bool))
    #         label_filled_relabeled = skimage.measure.label(label_filled).astype(np.uint64)

    #         print 'Working on z', z

    #         labels = range(len(Util.get_histogram(label_filled_relabeled)))#[1:] ### remove!!
    #         for l in labels:

    #             neighbors = MergeError.grab_neighbors(label_filled_relabeled, l)
    #             for n in neighbors:

    #                 upper_limit = 3

    #                 m = MergeError.create(image[z], prob[z], label_filled_relabeled, l, n, thumb)
    #                 while not m:

    #                     if upper_limit == 0:
    #                       #print "Upper limit reached."
    #                       break 

    #                     m = MergeError.create(image[z], prob[z], label_filled_relabeled, l, n, thumb)
    #                     upper_limit -= 1

    #                 if not m:
    #                   continue

    #                 yield m

    #                 if flip:
    #                     yield m.fliplr()
    #                     yield m.flipud()

    #                 if rotate:
    #                     yield m.rotate(90)
    #                     yield m.rotate(180)
    #                     m270 = m.rotate(270)
    #                     yield m270
    #                     if flip:
    #                         yield m270.fliplr()
    #                         yield m270.flipud()





    @staticmethod
    def generate(image, prob, label, n=10, thumb=False, rotate=True, flip=True, randomize_slice=False, randomize_label=False, max_per_slice=-1, fill_labels=True):
        '''
        '''

        # run through all slices

        if randomize_slice:
            z_s = np.arange(image.shape[0])
            np.random.shuffle(z_s)
        else:
            z_s = range(image.shape[0]) 

        for z in z_s:

            if fill_labels:
                # fill segmentation
                print 'Filling segmentation'
                label_zeros = Util.threshold(label[z], 0)
                label_filled = Util.fill(label[z], label_zeros.astype(np.bool))
                label_filled_relabeled = skimage.measure.label(label_filled).astype(np.uint64)
            else:
                print 'Skip filling'
                label_filled_relabeled = label
                print label_filled_relabeled.shape

            print 'Working on z', z

            slice_counter = 0

            if randomize_label:
                labels = np.arange(len(Util.get_histogram(label_filled_relabeled)))
                np.random.shuffle(labels)
            else:
                labels = range(1,len(Util.get_histogram(label_filled_relabeled))) # we ignore background 0 which should not exist anyways



            for l in labels:

                if slice_counter >= max_per_slice:
                    continue

                neighbors = MergeError.grab_neighbors(label_filled_relabeled, l)
                for n in neighbors:

                    upper_limit = 3

                    s = MergeError.create(image[z], prob[z], label_filled_relabeled, l, n, thumb)
                    while not s:

                        if upper_limit == 0:
                          # print "Upper limit reached."
                          break 

                        s = MergeError.create(image[z], prob[z], label_filled_relabeled, l, n, thumb)
                        upper_limit -= 1

                    if not s:
                      continue

                    yield s
                    slice_counter += 1
                    if slice_counter >= max_per_slice:
                        continue                    

                    if flip:
                        yield s.fliplr()
                        slice_counter += 1
                        if slice_counter >= max_per_slice:
                            continue

                        yield s.flipud()
                        slice_counter += 1
                        if slice_counter >= max_per_slice:
                            continue                        

                    if rotate:
                        yield s.rotate(90)
                        slice_counter += 1
                        if slice_counter >= max_per_slice:
                            continue


                        yield s.rotate(180)
                        slice_counter += 1
                        if slice_counter >= max_per_slice:
                            continue


                        m270 = s.rotate(270)
                        yield m270
                        slice_counter += 1
                        if slice_counter >= max_per_slice:
                            continue


                        if flip:
                            yield m270.fliplr()
                            slice_counter += 1
                            if slice_counter >= max_per_slice:
                                continue


                            yield m270.flipud()
                            slice_counter += 1
                            if slice_counter >= max_per_slice:
                                continue                            



