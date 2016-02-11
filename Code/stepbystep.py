
import cv2
import glob
import numpy as np
import mahotas as mh
import os
import uuid
import tifffile as tif
from scipy import ndimage as nd
from scipy.misc import imrotate
import skimage.measure
from skimage import img_as_ubyte
import random
import cPickle as pickle
import time
import partition_comparison
import _metrics
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
GOLD_PATH = os.path.join(DATA_PATH,'gold/')
RHOANA_PATH = os.path.join(DATA_PATH,'rhoana/')
IMAGE_PATH = os.path.join(DATA_PATH,'input/')
PROB_PATH = os.path.join(DATA_PATH,'prob/')
PATCH_PATH = os.path.join(DATA_PATH,'test_rhoana/')


gold = mh.imread(GOLD_PATH+'ac3_labels_0070.tif')
rhoana = mh.imread(RHOANA_PATH+os.sep+"z=00000070.tif")
image = mh.imread(IMAGE_PATH+'ac3_input_0070.tif')
prob = mh.imread(PROB_PATH+'ac3_input_0070_syn.tif')


PATCHES_BUFFER = []

def grab_neighbors(array, label):

    thresholded_array = _metrics.Util.threshold(array, label)
    thresholded_array_dilated = mh.dilate(thresholded_array.astype(np.uint64))

    all_neighbors = np.unique(array[thresholded_array_dilated == thresholded_array_dilated.max()].astype(np.uint64))
    all_neighbors = np.delete(all_neighbors, np.where(all_neighbors == label))

    return all_neighbors

def get_border_center(border, border_yx):
    node = mh.center_of_mass(border)
    nodes = border_yx
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    border_center = border_yx[np.argmin(dist_2)]    

    return border_center
    
def grab_patch(image, prob, segmentation, l, n, patch_size=(75,75), skip_boundaries=False, sample_rate=1,debug=False):

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
        # print border_center[0]-border_bbox[0], border_center[1]-border_bbox[2]
                patch_centers.append(sample_point)


        # sample_count = 0
        # while sample_rate>sample_count:

        #     sample_count += 1
            
        #     borders_labeled = skimage.measure.label(borders)



        #     border_centers = []

        #     for b in range(1,borders_labeled.max()+1):
        #         border = _metrics.Util.threshold(borders_labeled, b)

        #         border_yx = indices = zip(*np.where(border==1))

        #         # border_center = get_border_center(border, border_yx)
        #         #print border_yx
        #         border_center = (border_yx[len(border_yx)/2][0], border_yx[len(border_yx)/2][1])
        #         print border_center[0]-border_bbox[0], border_center[1]-border_bbox[2]
        #         border_centers.append(border_center)

        #     # split the borders into two
        #     for c in border_centers:

        #         if c[0]-1 > 0:
        #             borders[c[0]-1, c[1]] = 0
        #         if c[0]+1 < borders.shape[0]:
        #             borders[c[0]+1, c[1]] = 0
        #         borders[c[0], c[1]] = 0
        #         if c[1]-1 > 0:
        #             borders[c[0], c[1]-1] = 0
        #         if c[1]+1 < borders.shape[1]:
        #             borders[c[0], c[1]+1] = 0

                
                
        #     # recalculate the labeled borders
        #     borders_labeled = skimage.measure.label(borders)
        
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
#     # print 'bmax',borders_labeled.max()
    
#     # check if we need multiple patches because of multiple borders
#     for b in range(1,borders_labeled.max()+1):
# #         print 'patch for border', b
        
#         border = _metrics.Util.threshold(borders_labeled, b)

#         border_yx = indices = zip(*np.where(border==1))

#         # fault check if no border is found
#         if len(border_yx) == 0:
#             print 'no border', l, n, 'border', b
#             continue

#         #
#         # calculate border center properly
#         #
#         border_center = get_border_center(border, border_yx)
        

# #         # now we have the border center, this will be our first patch
# #         patch_centers = []
# #         patch_centers.append(border_center)
        
# #         # we also want to have to more patches
# #         border[border_center[0]-1, border_center[1]] = 0
# #         border[border_center[0]+1, border_center[1]] = 0
# #         border[border_center[0], border_center[1]] = 0
# #         border[border_center[0], border_center[1]-1] = 0
# #         border[border_center[0], border_center[1]+1] = 0
# #         borders_labeled2 = skimage.measure.label(border)
        
# #         other_center1 = _metrics.Util.threshold(borders_labeled2, 1)
# #         other_border_yx1 = zip(*np.where(other_center1==1))
# #         other_center1 = get_border_center(other_center1, other_border_yx1)
# #         other_center2 = _metrics.Util.threshold(borders_labeled2, 2)
# #         other_border_yx2 = zip(*np.where(other_center2==1))
# #         other_center2 = get_border_center(other_center2, other_border_yx2)
        
# #         patch_centers.append(other_center1)
# #         patch_centers.append(other_center2)        
        
        
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
        array1 = _metrics.Util.threshold(segmentation, l).astype(np.uint8)
        # threshold for label2
        array2 = _metrics.Util.threshold(segmentation, n).astype(np.uint8)
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
    
    

def loop(image, prob, segmentation, sample_rate):
    labels = range(1,len(_metrics.Util.get_histogram(segmentation.astype(np.uint64))))

    patches = []
    
    for l in labels:

        neighbors = grab_neighbors(segmentation, l)

        for n in neighbors:
            p = grab_patch(image, prob, segmentation, l, n, patch_size=(75,75), skip_boundaries=False, sample_rate=sample_rate)
            if not p:
                continue
                
            patches += p
            
    return patches

def fill_matrix(val_fn, m, m_p, patches, store_patches):
    
    patch_grouper = {}
    for p in patches:
        # create key
        minlabel = min(p['l'], p['n'])
        maxlabel = max(p['l'], p['n'])
        key = str(minlabel)+'-'+str(maxlabel)

        if not key in patch_grouper:
            patch_grouper[key] = []

        patch_grouper[key] += [p]

        if store_patches:
            # print 'storing', minlabel, maxlabel
            if m_p[minlabel, maxlabel] == None:
                m_p[minlabel, maxlabel] = []

            m_p[minlabel, maxlabel] += [p]

            if m_p[maxlabel, minlabel] == None:
                m_p[maxlabel, minlabel] = []

            m_p[maxlabel, minlabel] += [p]


    # now average the probabilities
    for k in patch_grouper.keys():

        weights = []
        predictions = []

        for p in patch_grouper[k]:

            # calculate the border length based on the patch size
            bbox = p['bbox']
            valid_border_points = 0
            for c in p['border']:
                if c[0] >= bbox[0] and c[0] <= bbox[1]:
                    if c[1] >= bbox[2] and c[1] <= bbox[3]:
                        # valid border point
                        valid_border_points += 1



            weights.append(valid_border_points)
            predictions.append(test_patch(val_fn, p))

        if len(patch_grouper[k]) == 1:
            weights[0] = 1

        p_sum = 0
        w_sum = 0
        for i,w in enumerate(weights):
            # weighted arithmetic mean
            p_sum += w*predictions[i]
            w_sum += w

        p_sum /= w_sum

        patch_grouper[k] = p_sum
#        patch_grouper[k] = min(predictions)

    for p in patch_grouper.keys():
        l = int(p.split('-')[0])
        n = int(p.split('-')[1])
        
        m[l,n] = patch_grouper[p]
        m[n,l] = patch_grouper[p]
        
    #print 'merged', len(patches), 'patches into', len(patch_grouper.keys())
        
    return m, m_p

            
def setup_n():
    from test_cnn_vis import TestCNN
    t = TestCNN('7b76867e-c76a-416f-910a-7065e93c616a', 'patches_large2new')
    val_fn = t.run()
    
    return val_fn
            
def test_patch(val_fn, p):
    # print p['image'].shape
    # print p['prob'].shape
    # print p['overlap'].shape
    images = p['image'].reshape(-1, 1, 75, 75).astype(np.uint8)
    probs = p['prob'].reshape(-1, 1, 75, 75).astype(np.uint8)
    binary1s = p['binary1'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    binary2s = p['binary2'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    overlaps = p['overlap'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    targets = np.array([0], dtype=np.uint8)
    
    pred, err, acc = val_fn(images, probs, binary1s, binary2s, overlaps, targets)
            
    return pred[0][1]

def create_matrix(val_fn, segmentation, patches, store_patches = False):
    
    
    no_labels = len(_metrics.Util.get_histogram(segmentation.astype(np.uint64)))
    
    m = np.zeros((no_labels,no_labels), dtype=np.float)
    m[:,:] = -1 # all uninitialised

    if store_patches:
        m_p = np.empty((no_labels,no_labels), dtype=np.object)
    else:
        m_p = None
 
#     for p in patches:

#         prediction = test_patch(val_fn, p)
#         m[p['l'], p['n']] = prediction
#         m[p['n'], p['l']] = prediction

    m, m_p = fill_matrix(val_fn, m, m_p, patches, store_patches)

    return m, m_p

P8_10_bef = None
P8_10_aft = None

def merge(val_fn, image, seg, prob, m_old, m_p_old, mode='random', mode2='random', counter=0, sureness=1., sample_rate=1, pick_first_random=False, pick_first_random_threshold=0.85, store_patches=False):
    
    # find largest value in matrix
    largest_index = np.where(m_old==m_old.max())

    # if pick_first_random:
    other_large_ones = np.where(m_old>=pick_first_random_threshold)
        
    #print 'largest', largest_index
    # print 'others', other_large_ones

    picked = 0

    smallest = np.Infinity
    smallest_label = -1
    largest = 0
    largest_label = -1

    # different picking
    if mode == 'random':
        #print 'picking randomly'
        picked = np.random.randint(len(largest_index[0]))
    elif mode == 'smallest':
        #print 'picking smallest first'
        for i,label in enumerate(largest_index[0]):
          current_size = len(seg[seg == label])
          if current_size < smallest:
            smallest = current_size
            smallest_label = label
            picked = i
        #print 'label', smallest_label, 'is smallest'
       
    elif mode == 'largest':
        #print 'picking largest first'
        for i,label in enumerate(largest_index[0]):
          current_size = len(seg[seg == label])
          if current_size > largest:
            largest = current_size
            largest_label = label
            picked = i
        #print 'label', largest_label, 'is largest'


    
    l,n = (largest_index[0][picked], largest_index[1][picked])

    # now we have the merge pair, let's check if either one has candidates with
    # the same prob.
    others_l = np.where(m_old[l,:] == m_old.max())

    l_index = 0

    #print 'chose n=', n, 'so far..'

    smallest = np.Infinity
    smallest_label = -1
    smallest_index = -1
    largest = 0
    largest_label = -1
    largest_index = -1

    # different picking
    if mode2 == 'random':
        #print 'picking randomly'
        picked = np.random.randint(len(others_l[0]))
    elif mode2 == 'smallest':
        #print 'picking smallest first'
        for i,label in enumerate(others_l[0]):
          current_size = len(seg[seg == label])
          if current_size < smallest:
            smallest = current_size
            smallest_label = label
            smallest_index = i
        #print 'label n', smallest_label, 'is smallest'
        picked = smallest_index
    elif mode2 == 'largest':
        #print 'picking largest first'
        for i,label in enumerate(others_l[0]):
          current_size = len(seg[seg == label])
          if current_size > largest:
            largest = current_size
            largest_label = label
            largest_index = i
        #print 'label n', largest_label, 'is largest'
        picked = largest_index

    n = others_l[0][picked]
    #print 'now chose n=', n

    return perform_merge(val_fn, image, seg, prob, m_old, m_p_old, l, n, sample_rate, sureness, store_patches)



def merge_with_neighbor_check(val_fn, image, seg, prob, m_old, m_p_old, counter=0, sureness=1., sample_rate=1, pick_first_random=False, pick_first_random_threshold=0.85, store_patches=False):
    
    # find largest value in matrix
    largest_index = np.where(m_old==m_old.max())

    l,n = (largest_index[0][0], largest_index[1][0])

    print 'merging', l, n, m_old.max()
    
    # check matrix for both l and n if other merges are also high
    others_l = np.where(m_old[l,:] == m_old.max())
    others_n = np.where(m_old[n,:] == m_old.max())

    print others_l, others_n

    # return None


    return perform_merge(val_fn, image, seg, prob, m_old, m_p_old, l, n, sample_rate, sureness, store_patches)



def perform_merge(val_fn, image, seg, prob, m_old, m_p_old, l, n, sample_rate, sureness=1., store_patches=False):

    
    out = np.array(seg)
    m = np.array(m_old)
    m_p = np.array(m_p_old)
    
    patches = []    
    
    old_l = l
    old_n = n


    # merge these labels
    # print 'merging', l, n
    out[out == l] = n
    # set matrix as merged for this entry
    m[l,:] = -2
    m[:,l] = -2
    if store_patches:
        m_p[l,:] = None
        m_p[:,l] = None
    
    # grab neighbors of l
    old_neighbors = grab_neighbors(seg, l)
    # get patches for l and n_l
    for n in old_neighbors:
        neighbors = grab_neighbors(out, n)
        for k in neighbors:
            
            # check if this is still a valid combination
            if m[n,k] == -2:
                # nope!
                print 'not valid', n, k
                continue
            
            p = grab_patch(image, prob, out, n, k, patch_size=(75,75), skip_boundaries=False, sample_rate=sample_rate)
            if not p:
#                 print 'problem', n, k
                continue

            patches += p
        
    # print 'recomputed', len(patches), 'patches'

#     for p_ in patches:
#         print p_['l'], p_['n'], round(test_patch(val_fn, p_),2)
    

#     print 'bef fill', m[8,10], m[10,8]
    
    m, m_p = fill_matrix(val_fn, m, m_p, patches, store_patches)

#     print 'aft fill', m[8,10], m[10,8]
    
    return m, m_p, out, patches, old_l, old_n



def propagate_max_overlap(rhoana, gold):

    out = np.array(rhoana)
    
    rhoana_labels = _metrics.Util.get_histogram(rhoana.astype(np.uint64))
    
    for l,k in enumerate(rhoana_labels):
        if l == 0 or k==0:
            # ignore 0 since rhoana does not have it
            continue
        values = gold[rhoana == l]
        largest_label = _metrics.Util.get_largest_label(values.astype(np.uint64))
    
        out[rhoana == l] = largest_label # set the largest label from gold here
    
    return out
        
    
    
    
    
def create_spliterrors(image, seg, n = 1):
    
    patch_size = 75#(75,75)
    
    
    big_out = np.array(seg)
    #out = np.array(seg[patch_size/2:-patch_size/2,patch_size/2:-patch_size/2])
    out = np.array(seg)
    seg_labels = _metrics.Util.get_histogram(out.astype(np.uint64))
    print seg_labels
    
    new_label = len(seg_labels)
    
    for l,k in enumerate(seg_labels):
        if l == 0 or k<100:
            # ignore 0 since seg does not have it
#             print 'ignore',l
            continue

        # if l % 2 == 0:
        #     continue
    
        for r in range(n):

    #         print l
        
    #         # ignore boundaries
    #         bbox = mh.labeled.bbox(out)[l]
    #         if bbox[0] <= 33:
    #             continue
    #         if bbox[1] >= out.shape[0]-33:
    #             continue
    #         if bbox[2] <= 33:
    #             continue
    #         if bbox[3] >= out.shape[1]-33:
    #             continue
        
    #         split_label = split(np.array(image[patch_size/2:-patch_size/2,patch_size/2:-patch_size/2]), out, l)
            split_label = split(np.array(image), out, l)
            
            #split_labels = len(_metrics.Util.get_histogram(split_label.astype(np.uint64)))
            firstlabel = split_label.max()
            secondlabel = firstlabel - 1
            
            split_label[split_label == firstlabel] = new_label
            split_label[split_label == secondlabel] = new_label + 1
            
            #print firstlabel, secondlabel, new_label
            new_label += 2

            
            out[out == l] = split_label[split_label != 0]
            l = new_label-2
    
    #big_out[patch_size/2:-patch_size/2,patch_size/2:-patch_size/2] = out
    big_out = out
    return big_out
    

def split(image, array, label):
    '''
    '''

    large_label = _metrics.Util.threshold(array, label)
#     imshow(large_label)
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


    large_label = np.zeros(large_label.shape, dtype=np.uint64)
    large_label[label_bbox[0]:label_bbox[1], label_bbox[2]:label_bbox[3]] = ws
    return large_label      
    
