import mahotas
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import glob
import os

from cnn import *
from mlp import HiddenLayer, MLP, rectified_linear
from generateTrainValTestData import generate_experiment_data_supervised, shared_dataset, normalizeImage, stupid_map_wrapper
from scipy.ndimage.interpolation import shift

import time

from whole_image_classification import *

    
if __name__ == '__main__':
    rng = numpy.random.RandomState(929292)

    import mahotas
    import matplotlib.pyplot as plt
    #CPU
    pathPrefix = '/home/d/data/S1_RDExtendLeft/orig_images/'#'/Users/d/Projects/dojo_data_vis2014/images/'#'/Volumes/DATA1/EMQM_DATA/ac3x75/input/'
    img_search_string_grayImages = pathPrefix + '*.png'
    img_files = sorted( glob.glob( img_search_string_grayImages ) )

    for image_path in img_files:

        if os.path.exists(image_path.replace('.png', '_syn.tif')):
          continue
        print image_path
        image = mahotas.imread(image_path)
        imageSize = image.shape[0]#1024
        image = image[0:imageSize,0:imageSize]
        
        start_time = time.clock()
        image = normalizeImage(image) - 0.5
        
        #GPU
        image_shared = theano.shared(np.float32(image))
        image_shared = image_shared.reshape((1,1,imageSize,imageSize))
        
        x = T.matrix('x')
        
        classifier = CNN(input=x, batch_size=imageSize, patchSize=65, rng=rng, nkerns=[48,48,48], kernelSizes=[5,5,5], hiddenSizes=[200], fileName='thin_membranes_exhaustive_training_0.01.pkl')
        
        fragments = [image_shared]
        
        print "Convolutions"
        
        for clayer in classifier.convLayers:
            newFragments = []
            for img_sh in fragments:
                # CPU???
                convolved_image = get_convolution_output(image_shared=img_sh, clayer=clayer)
                #GPU
                output = get_max_pool_fragments(convolved_image, clayer=clayer)
                newFragments.extend(output)
                
            fragments = newFragments

        #### now the hidden layer
    
        print "hidden layer"
        
        hidden_fragments = []
        
        for fragment in fragments:
            hidden_out = get_hidden_output(image_shared=fragment, hiddenLayer=classifier.mlp.hiddenLayers[0], nHidden=200, nfilt=classifier.nkerns[-1])
            hidden_fragments.append(hidden_out)

            #### and the missing log reg layer

        print "logistic regression layer"
        
        final_fragments = []
        for fragment in hidden_fragments:
            logreg_out = get_logistic_regression_output(image_shared=fragment, logregLayer=classifier.mlp.logRegressionLayer)
            logreg_out = logreg_out.eval()
            logreg_out = logreg_out[0,:,:]
            final_fragments.append(logreg_out)

            
        prob_img = np.zeros(image.shape)
        
        offsets_tmp = np.array([[0,0],[0,1],[1,0],[1,1]])
        
        if len(classifier.convLayers)>=1:
            offsets = offsets_tmp
            
        if len(classifier.convLayers)>=2:
            offset_init_1 = np.array([[0,0],[0,1],[1,0],[1,1]])
            offset_init_2 = offset_init_1 * 2
            
            offsets = np.zeros((4,4,2))
            for o_1 in range(4):
                for o_2 in range(4):
                    offsets[o_1,o_2] = offset_init_1[o_1] + offset_init_2[o_2]
                    
            offsets = offsets.reshape((16,2))

        if len(classifier.convLayers)>=3:
            offset_init_1 = offsets.copy()
            offset_init_2 =  np.array([[0,0],[0,1],[1,0],[1,1]]) * 4
            
            offsets = np.zeros((16,4,2))
            for o_1 in range(16):
                for o_2 in range(4):
                    offsets[o_1,o_2] = offset_init_1[o_1] + offset_init_2[o_2]
                    
            offsets = offsets.reshape((64,2))

        offset_jumps = np.int16(np.sqrt(len(offsets)))
        for f, o in zip(final_fragments, offsets):
            prob_size = prob_img[o[0]::offset_jumps,o[1]::offset_jumps].shape
            f_s = np.zeros(prob_size)
            f_s[:f.shape[0], :f.shape[1]] = f.copy()
            prob_img[o[0]::offset_jumps,o[1]::offset_jumps] = f_s
            
        total_time = time.clock() - start_time
        print "This took %f seconds." % (total_time)
        
        prob_img = shift(prob_img,(32,32))
        
        mahotas.imsave(image_path[:-4] + '_syn.tif', 
                       np.uint8((1-prob_img)*255))
        

    
 
    
