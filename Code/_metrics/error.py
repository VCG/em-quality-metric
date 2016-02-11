import cPickle as pickle
import cv2
import mahotas as mh
import numpy as np
import os
import random
import skimage.measure
import uuid

from util import Util

class Error(object):
    '''
    '''
    def __init__(self):
        self._meta = None
        self._thumb = None
        self._has_thumb = False
        
    def store(self, folder):
        if not self._meta:
            # this is no good stuff
            return -1
        
        f = folder + os.sep + self._meta['id']
        #np.savez(f, self._meta)
        pickle.dump(self._meta, open(f+'.p','wb'))
        cv2.imwrite(f+'.tif', self._thumb)
        
        return self._meta['id']
    
    @staticmethod
    def split(image, array, label):
        '''
        '''

        large_label = Util.threshold(array, label)

        label_bbox = mh.bbox(large_label)
        label = large_label[label_bbox[0]:label_bbox[1], label_bbox[2]:label_bbox[3]]
        image = image[label_bbox[0]:label_bbox[1], label_bbox[2]:label_bbox[3]]

        #
        # smooth the image
        #
        image = mh.gaussian_filter(image, 3.5)

        grad_x = np.gradient(image)[0]
        grad_y = np.gradient(image)[1]
        grad = np.add(np.abs(grad_x), np.abs(grad_y))
        #grad = np.add(np.abs(grad_x), np.abs(grad_y))
        grad -= grad.min()
        grad /= grad.max()
        grad *= 255
        grad = grad.astype(np.uint8)
        #imshow(grad)

        # we need 4 labels as output
        max_label = 0
        #while max_label!=3:

        coords = zip(*np.where(label==1))

        seed1 = random.choice(coords)
        seed2 = random.choice(coords)
        seeds = np.zeros(label.shape, dtype=np.uint64)
        seeds[seed1] = 1
        seeds[seed2] = 2
#         imshow(seeds)
        for i in range(10):
            seeds = mh.dilate(seeds)

        ws = mh.cwatershed(grad, seeds)
        ws[label==0] = 0

        ws_relabeled = skimage.measure.label(ws.astype(np.uint64))
        max_label = ws_relabeled.max()
        #print max_label

        large_label = np.zeros(large_label.shape, dtype=np.uint64)
        large_label[label_bbox[0]:label_bbox[1], label_bbox[2]:label_bbox[3]] = ws
        return large_label        
        
    def create_thumb(self, image, merge_output):
        '''
        '''
        if not merge_output:
            return None
        
        bbox = merge_output['bbox']

        image_bbox = image[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]

        output = np.zeros((image_bbox.shape[0], image_bbox.shape[1], 3),dtype=np.uint8)

        output[:,:,0] = image_bbox
        output[:,:,1] = image_bbox
        output[:,:,2] = image_bbox

        output[merge_output['label1'] == 1] = [255,0,0]
        output[merge_output['label2'] == 1] = [0,255,0]
        output[merge_output['overlap'] == 1] = [0,0,255]

        for i,c in enumerate(merge_output['border']):

            # filter all border points which are outside of the bounding box
            if c[0] < bbox[0] or c[0] > bbox[1]:
                continue

            if c[1] < bbox[2] or c[1] > bbox[3]:
                continue

            output[c[0]-bbox[0], c[1]-bbox[2]] = [255,0,0]

        return output        

    def create_tensors(self, image, prob, merge_output):
        '''
        '''
        if not merge_output:
            return None
        
        bbox = merge_output['bbox']

        image_bbox = image[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
        bin1 = merge_output['label1']
        bin2 = merge_output['label2']
        prob = prob[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
        overlap = merge_output['overlap']

        return [image_bbox, bin1, bin2, prob, overlap]

        # we return
        # 1) image
        # 2) binary mask 1
        # 3) binary mask 2
        # 4) prob
        # 5) overlap

        

    @staticmethod
    def analyze_border(image, prob, segmentation, l, n, patch_size=(75,75), skip_boundaries=False, sample_rate=1,debug=False):
        #self, image, prob, array, original_label, label1, label2, overlap=10, patch_size=(75,75)):

        borders = mh.labeled.border(segmentation, l, n)

        #
        # treat interrupted borders separately
        #
        borders_labeled = skimage.measure.label(borders)

        # print borders_labeled.max()
        # for b in range(1,borders_labeled.max()+1):
            # borders = _metrics.Util.threshold(borders_labeled, b)



        border_bbox = mh.bbox(borders)


        patch_centers = []
        border_yx = indices = zip(*np.where(borders==1))

        # always sample the middle point
        border_center = (border_yx[len(border_yx)/(2)][0], border_yx[len(border_yx)/(2)][1])
        patch_centers.append(border_center)


        if sample_rate > 1 or sample_rate == -1:
            if sample_rate > len(border_yx) or sample_rate==-1:
                samples = 1
            else:
                samples = len(border_yx) / sample_rate

            for i,s in enumerate(border_yx):
                
                if i % samples == 0:

                    sample_point = s
            
                    patch_centers.append(sample_point)
            
        borders_w_center = np.array(borders.astype(np.uint8))

        for i,c in enumerate(patch_centers):
            


            borders_w_center[c[0],c[1]] = 10*(i+1)
            # print 'marking', c, borders_w_center.shape

        if debug:
            fig = plt.figure(figsize=(5,5))
            fig.text(0,1,'\n\n\n\n\nAll borders '+str(l)+','+str(n))#+'\n\n'+str(np.round(_matrices[u], 2)))
            plt.imshow(borders_labeled[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]], interpolation='nearest')
            fig = plt.figure(figsize=(5,5))
            fig.text(0,1,'\n\n\n\n\nWith center(s) '+str(l)+','+str(n))#+'\n\n'+str(np.round(_matrices[u], 2)))
            plt.imshow(borders_w_center[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]], interpolation='nearest')#, cmap='ocean')
        
        
        patches = []
            
            
        for i,c in enumerate(patch_centers):

            
    #         for border_center in patch_centers:

            # check if border_center is too close to the 4 edges
            new_border_center = [c[0], c[1]]

            if new_border_center[0] < patch_size[0]/2:
                # print 'oob1', new_border_center
                # return None
                continue
            if new_border_center[0]+patch_size[0]/2 >= segmentation.shape[0]:
                # print 'oob2', new_border_center
                # return None
                continue
            if new_border_center[1] < patch_size[1]/2:
                # print 'oob3', new_border_center
                # return None
                continue
            if new_border_center[1]+patch_size[1]/2 >= segmentation.shape[1]:
                # print 'oob4', new_border_center
                # return None
                continue
            # print new_border_center, patch_size[0]/2, border_center[0] < patch_size[0]/2

            # continue


            bbox = [new_border_center[0]-patch_size[0]/2, 
                    new_border_center[0]+patch_size[0]/2,
                    new_border_center[1]-patch_size[1]/2, 
                    new_border_center[1]+patch_size[1]/2]

            ### workaround to not sample white border of probability map
            if skip_boundaries:
                if bbox[0] <= 33:
                    # return None
                    # print 'ppb'
                    continue
                if bbox[1] >= segmentation.shape[0]-33:
                    # return None
                    # print 'ppb'
                    continue
                if bbox[2] <= 33:
                    # return None
                    # print 'ppb'
                    continue
                if bbox[3] >= segmentation.shape[1]-33:
                    # return None
                    # print 'ppb'
                    continue

            

            # threshold for label1
            array1 = Util.threshold(segmentation, l).astype(np.uint8)
            # threshold for label2
            array2 = Util.threshold(segmentation, n).astype(np.uint8)
            merged_array = array1 + array2


            

            # dilate for overlap
            dilated_array1 = np.array(array1)
            dilated_array2 = np.array(array2)
            for o in range(10):
                dilated_array1 = mh.dilate(dilated_array1.astype(np.uint64))
                dilated_array2 = mh.dilate(dilated_array2.astype(np.uint64))
            overlap = np.logical_and(dilated_array1, dilated_array2)
            overlap[merged_array == 0] = 0

            

            output = {}
            output['id'] = str(uuid.uuid4())
            output['image'] = image[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
            
            output['prob'] = prob[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
            
            output['l'] = l
            output['n'] = n
            output['bbox'] = bbox
            output['border'] = border_yx
            output['border_center'] = new_border_center
            output['binary1'] = array1[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
            output['binary2'] = array2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
            output['overlap'] = overlap[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
            output['borders_labeled'] = borders_labeled[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]]
            output['borders_w_center'] = borders_w_center[border_bbox[0]:border_bbox[1], border_bbox[2]:border_bbox[3]]

            patches.append(output)
            

        return patches


        
        # # threshold for label1
        # array1 = Util.threshold(array, label1).astype(np.uint8)
        # # threshold for label2
        # array2 = Util.threshold(array, label2).astype(np.uint8)

        # # add them
        # merged_array = array1 + array2

        # bbox_y1y2x1x2 = mh.bbox(merged_array)
        # bbox = bbox_y1y2x1x2        

        
        # # dilate for overlap
        # dilated_array1 = np.array(array1)
        # dilated_array2 = np.array(array2)
        # for o in range(overlap):
        #     dilated_array1 = mh.dilate(dilated_array1.astype(np.uint64))
        #     dilated_array2 = mh.dilate(dilated_array2.astype(np.uint64))

        # #dilated_sum = np.zeros(merged_array.shape)
        # overlap = np.logical_and(dilated_array1, dilated_array2)
        # overlap[merged_array == 0] = 0

        # border = mh.labeled.border(array, int(label1), int(label2))
        
        # border_yx = indices = zip(*np.where(border==True))

        # # fault check if no border is found
        # if len(border_yx) < 2:
        #     # print 'no border'
        #     return None

        # #
        # #
        # # check if there is more than one border
        # empty = np.zeros((bbox[1]-bbox[0], bbox[3]-bbox[2]))

        # for c in border_yx:
        #     empty[c[0]-bbox[0],c[1]-bbox[2]] = 1
            
        # empty_labeled = skimage.measure.label(empty)
        # #mh.imsave('/Volumes/DATA1/test.tif', empty_labeled.astype(np.uint8)*255)
        # if empty_labeled.max() > 1:
        #     # print 'more than 1 border', empty_labeled.max()
        #     print Util.get_largest_label(empty_labeled.astype(np.uint8), True)
        #     return None
        # #
        # #
        # #
        
        # #
        # # check if there the labels are enclosing each other
        # #
        # array1_no_holes = np.all(mh.close_holes(array1[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(np.bool)) == array1[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(np.bool))
        # array2_no_holes = np.all(mh.close_holes(array2[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(np.bool)) == array2[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(np.bool))        
        # if (not array1_no_holes or not array2_no_holes):
        #     # this is no good
        #     # print 'enclosing'
        #     return None
        
        
        # #
        # # calculate border center properly
        # #
        # border_normalized = border_yx#[(u-bbox[0],v-bbox[2]) for u,v in border_yx]
        # empty = np.zeros(merged_array.shape,dtype=np.bool)
        # for c in border_normalized:
        #     empty[c[0],c[1]] = 1
        # node = mh.center_of_mass(empty)
        # nodes = border_normalized
        # nodes = np.asarray(nodes)
        # deltas = nodes - node
        # dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        # border_center = border_normalized[np.argmin(dist_2)]
        
        # # check if border_center is too close to the 4 edges
        # new_border_center = [border_center[0], border_center[1]]
        # if border_center[0] < patch_size[0]/2:
        #     new_border_center[0] = patch_size[0]/2
        #     # print 'too close'
        #     return None
        # if border_center[0]+patch_size[0]/2 >= merged_array.shape[0]:
        #     new_border_center[0] = merged_array.shape[0] - patch_size[0]/2 - 1
        #     # print 'too close'
        #     return None
        # if border_center[1] < patch_size[1]/2:
        #     new_border_center[1] = patch_size[1]/2
        #     # print 'too close'
        #     return None
        # if border_center[1]+patch_size[1]/2 >= merged_array.shape[1]:
        #     new_border_center[1] = merged_array.shape[1] - patch_size[1]/2 - 1
        #     # print 'too close'
        #     return None

        
        # bbox = [new_border_center[0]-patch_size[0]/2, 
        #         new_border_center[0]+patch_size[0]/2,
        #         new_border_center[1]-patch_size[1]/2, 
        #         new_border_center[1]+patch_size[1]/2]


        # ### workaround to not sample white border of probability map
        # if bbox[0] <= 33:
        #     # print 'prob'
        #     return None
        # if bbox[1] >= merged_array.shape[0]-33:
        #     # print 'prob'
        #     return None
        # if bbox[2] <= 33:
        #     # print 'prob'
        #     return None
        # if bbox[3] >= merged_array.shape[1]-33:
        #     # print 'prob'
        #     return None



        # output = {}
        # output['id'] = str(uuid.uuid4())
        # output['image'] = image[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
        # output['prob'] = prob[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1]
        # output['label'] = original_label
        # output['bbox'] = bbox
        # output['border'] = border_yx
        # output['border_center'] = new_border_center
        # output['merge'] = merged_array[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
        # output['label1'] = array1[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
        # output['label2'] = array2[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)
        # output['overlap'] = overlap[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1].astype(np.bool)

        # return output        
        
        