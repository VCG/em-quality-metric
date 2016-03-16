import ast
import copy
import numpy as np
import os
import sys
import theano
import theano.tensor as T
import time
import pickle
import glob

import lasagne


class CNN(object):


  def __init__(self, network_id, patches='patches_large2_vis', inputs=['image', 'prob', 'binary', 'border_overlap'], patch_size=(75,75)):
    '''
    '''
    #self.initialize()
    self._dbg = None

    self._DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
    if not os.path.isdir(self._DATA_PATH):
      self._DATA_PATH = '/n/regal/pfister_lab/haehn/ac3x75/'
    self._PATCH = patches
    self._PATCH_PATH = os.path.join(self._DATA_PATH,self._PATCH+'/')
    self._NETWORK_ID = network_id
    self._OUTPUT_PATH = '/Volumes/DATA1/split_cnn'
    if not os.path.isdir(self._OUTPUT_PATH):
      self._OUTPUT_PATH = '/n/regal/pfister_lab/haehn/split_cnn'
    self._RESULTS_PATH = os.path.join(self._OUTPUT_PATH, self._PATCH, self._NETWORK_ID)
    self._BATCH_SIZE = -1


    self._inputs = inputs
    self._patch_size = patch_size

    self._epochs = -1
    self._test_loss = -1
    self._test_acc = -1
    self._configuration = None

    self._val_fn = self.run()

  def test_patch(self, p):

    patch_reshaped = {
      'image': p['image'].reshape(-1, 1, self._patch_size[0], self._patch_size[1]),
      'prob': p['prob'].reshape(-1, 1, self._patch_size[0], self._patch_size[1]),
      'binary': p['binary'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
      'merged_array': p['merged_array'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
      'binary1': p['binary1'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
      'binary2': p['binary2'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,      
      'dyn_obj': p['dyn_obj'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
      'dyn_bnd': p['dyn_bnd'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
      'dyn_bnd_dyn_obj': p['dyn_bnd_dyn_obj'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
      'overlap': p['overlap'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
      'border_overlap': p['border_overlap'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
      'larger_border_overlap': p['larger_border_overlap'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255 
    }



    # print p['image'].shape
    # print p['prob'].shape
    # print p['overlap'].shape
    # images = p['image'].reshape(-1, 1, 75, 75).astype(np.uint8)
    # probs = p['prob'].reshape(-1, 1, 75, 75).astype(np.uint8)
    # binary1s = p['binary1'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    # # binary2s = p['binary2'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    # overlaps = p['overlap'].reshape(-1, 1, 75, 75).astype(np.uint8)*255
    targets = np.array([0], dtype=np.uint8)
    
    test_values = []
    for i in self._inputs:
      test_values.append(patch_reshaped[i])
    test_values.append(targets)

    pred, err, acc = self._val_fn(*test_values)#images, probs, binary1s, overlaps, targets)
            
    return pred[0][1]


  def load_dataset(self, path=None, testfile='vis.npz', targetfile='vis_targets.npz'):


      if path == None:
        path = self._PATCH_PATH


      test = np.load(self._PATCH_PATH+'test.npz')
      test_targets = np.load(self._PATCH_PATH+'test_targets.npz')

      #
      # we also normalize all binary images as uint8
      #
      test = {
        'image': test['image'].reshape(-1, 1, self._patch_size[0], self._patch_size[1]),
        'prob': test['prob'].reshape(-1, 1, self._patch_size[0], self._patch_size[1]),
        'binary': test['binary'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
        'merged_array': test['merged_array'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
        'dyn_obj': test['dyn_obj'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
        'dyn_bnd': test['dyn_bnd'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
        'border_overlap': test['border_overlap'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255,
        'larger_border_overlap': test['larger_border_overlaps'].astype(np.uint8).reshape(-1, 1, self._patch_size[0], self._patch_size[1])*255
      }

      test_targets = test_targets['targets'].astype(np.uint8)

      #val = np.load(PATCH_PATH+'val.npz')
      #val_targets = np.load(PATCH_PATH+'val_targets.npz')

      return test, test_targets


  def load_network(self):
    '''
    '''
    network_file = sorted(glob.glob(self._RESULTS_PATH + os.sep + 'network*.p'))
    print 'Loading', network_file[-1]
    with open(network_file[-1], 'rb') as f:
        n = pickle.load(f)

    # print n

    self._epochs = network_file[-1].split('_')[-1].replace('.p','')

    stats_file = glob.glob(self._RESULTS_PATH + os.sep + 'final_test_*.txt')
    if len(stats_file) > 0:
      test_loss = os.path.basename(stats_file[0]).split('___')[0].replace('final_test_loss_','')
      self._test_loss = test_loss
      test_acc = os.path.basename(stats_file[0]).split('___')[1].replace('test_acc_','').replace('.txt','')
      self._test_acc = test_acc


    return n

  def load_configuration(self):
    '''
    '''
    config_file = glob.glob(self._RESULTS_PATH + os.sep + 'configuration.txt')
    print 'Loading', config_file

    with open(config_file[0], 'r') as f:
      c = f.readlines()

    

    c = ast.literal_eval(c[0])
    self._configuration = c

    return c

  def run(self, num_epochs=5):
    '''
    '''
    # Load the dataset
    # print("Loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset()
    # X_test, y_test = self.load_dataset()
    # X_val = X_test = X_train
    # y_val = y_test = y_train

    # Prepare Theano variables for inputs and targets
    # image_var = T.tensor4('image')
    # prob_var = T.tensor4('prob')
    # binary1_var = T.tensor4('binary1')
    # binary2_var = T.tensor4('binary2')
    # overlap_var = T.tensor4('overlap')
    target_var = T.ivector('targets')

    # layers = self.build(image_var, prob_var, binary1_var, binary2_var, overlap_var, self._THIRD_CONV_LAYER)
    # network = layers['dense']['network']

    config = self.load_configuration()
    self._BATCH_SIZE = config['batchsize']
    layers = self.load_network()
    # image_var = layers['image']['input_layer'].input_var
    # prob_var = layers['prob']['input_layer'].input_var
    # binary1_var = layers['binary1']['input_layer'].input_var
    # # binary2_var = layers['binary2']['input_layer'].input_var
    # overlap_var = layers['overlap']['input_layer'].input_var




    network = layers['dense']['network']

    theano_vars = []
    for i in self._inputs:
      theano_vars.append(layers[i]['input_layer'].input_var)
    target_var = T.ivector('targets')

    theano_function_vars = theano_vars + [target_var]    

    # # Create a loss expression for training, i.e., a scalar objective we want
    # # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # prediction = lasagne.layers.get_output(network)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # loss = loss.mean()
    # # We could add some weight decay as well here, see lasagne.regularization.

    # # Create update expressions for training, i.e., how to modify the
    # # parameters at each training step. Here, we'll use Stochastic Gradient
    # # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    # params = lasagne.layers.get_all_params(network, trainable=True)
    # # updates = lasagne.updates.nesterov_momentum(
    # #         loss, params, learning_rate=self._LEARNING_RATE, momentum=self._MOMENTUM)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    # train_fn = theano.function([image_var, prob_var, binary1_var, binary2_var, overlap_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    # val_fn = theano.function([image_var, prob_var, binary1_var, overlap_var, target_var], [test_prediction, test_loss, test_acc])
    val_fn = theano.function(theano_function_vars, [test_prediction, test_loss, test_acc])

    return val_fn


    # # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in self.iterate_minibatches(X_test, y_test, self._BATCH_SIZE, shuffle=False):
    #     # inputs, targets = batch
    #     # err, acc = val_fn(inputs, inputs, inputs, inputs, inputs, targets)
    #     images, probs, binary1s, binary2s, overlaps, targets = batch
    #     pred, err, acc = val_fn(images, probs, binary1s, binary2s, overlaps, targets)        
    #     print pred
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))

    # # self._test_loss.append(test_err / test_batches)
    # # self._test_acc.append(test_acc / test_batches * 100)

    # return layers




















  # ############################# Batch iterator ###############################
  # This is just a simple helper function iterating over training data in
  # mini-batches of a particular size, optionally in random order. It assumes
  # data is available as numpy arrays. For big datasets, you could load numpy
  # arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
  # own custom data iteration function. For small datasets, you can also copy
  # them to GPU at once for slightly improved performance. This would involve
  # several changes in the main program, though, and is not demonstrated here.

  def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
      assert len(inputs['image']) == len(targets)
      assert len(inputs['prob']) == len(targets)
      assert len(inputs['binary']) == len(targets)
      # assert len(inputs['binary2']) == len(targets)
      assert len(inputs['border_overlap']) == len(targets)
      if shuffle:
          indices = np.arange(len(inputs))
          np.random.shuffle(indices)
      for start_idx in range(0, len(inputs['image']) - batchsize + 1, batchsize):
          if shuffle:
              excerpt = indices[start_idx:start_idx + batchsize]
          else:
              excerpt = slice(start_idx, start_idx + batchsize)

          minibatches = []
          for i in self._inputs:
            minibatches.append(inputs[i][excerpt])
          minibatches.append(targets[excerpt])

          yield minibatches

  # def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
  #     assert len(inputs['image']) == len(targets)
  #     assert len(inputs['prob']) == len(targets)
  #     assert len(inputs['binary1']) == len(targets)
  #     # assert len(inputs['binary2']) == len(targets)
  #     assert len(inputs['overlap']) == len(targets)
  #     if shuffle:
  #         indices = np.arange(len(inputs))
  #         np.random.shuffle(indices)
  #     for start_idx in range(0, len(inputs['image']) - batchsize + 1, batchsize):
  #         if shuffle:
  #             excerpt = indices[start_idx:start_idx + batchsize]
  #         else:
  #             excerpt = slice(start_idx, start_idx + batchsize)
  #         yield inputs['image'][excerpt], inputs['prob'][excerpt], inputs['binary1'][excerpt], inputs['overlap'][excerpt], inputs['bbox'][excerpt], inputs['border'][excerpt], targets[excerpt]

  def visualize_filters(self, layer):

    # print layer

    W = layer.W.get_value()
    # print W

    shape = W.shape
    W_vis = np.reshape(W, (shape[0], shape[2]*shape[3]))

    return self.tile_raster_images(W_vis, (shape[2],shape[3]))

  def scale_to_unit_interval(self, ndar, eps=1e-8):
      """ Scales all values in the ndarray ndar to be between 0 and 1 """
      ndar = np.float32(ndar.copy())
      ndar -= ndar.min()
      ndar *= 1.0 / (ndar.max() + eps)
      return ndar

  def tile_raster_images(self, X, img_shape=(5,5), tile_shape=(10,10), tile_spacing=(0, 0),
                         scale_rows_to_unit_interval=True,
                         output_pixel_vals=True):
      """
      Transform an array with one flattened image per row, into an array in
      which images are reshaped and layed out like tiles on a floor.

      This function is useful for visualizing datasets whose rows are images,
      and also columns of matrices for transforming those rows
      (such as the first layer of a neural net).

      :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
      be 2-D ndarrays or None;
      :param X: a 2-D array in which every row is a flattened image.

      :type img_shape: tuple; (height, width)
      :param img_shape: the original shape of each image

      :type tile_shape: tuple; (rows, cols)
      :param tile_shape: the number of images to tile (rows, cols)

      :param output_pixel_vals: if output should be pixel values (i.e. int8
      values) or floats

      :param scale_rows_to_unit_interval: if the values need to be scaled before
      being plotted to [0,1] or not


      :returns: array suitable for viewing as an image.
      (See:`PIL.Image.fromarray`.)
      :rtype: a 2-d array with same dtype as X.

      """

      assert len(img_shape) == 2
      assert len(tile_shape) == 2
      assert len(tile_spacing) == 2

      # The expression below can be re-written in a more C style as
      # follows :
      #
      # out_shape    = [0,0]
      # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
      #                tile_spacing[0]
      # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
      #                tile_spacing[1]
      out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                          in zip(img_shape, tile_shape, tile_spacing)]

      if isinstance(X, tuple):
          assert len(X) == 4
          # Create an output numpy ndarray to store the image
          if output_pixel_vals:
              out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                      dtype='uint8')
          else:
              out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                      dtype=X.dtype)

          #colors default to 0, alpha defaults to 1 (opaque)
          if output_pixel_vals:
              channel_defaults = [0, 0, 0, 255]
          else:
              channel_defaults = [0., 0., 0., 1.]

          for i in xrange(4):
              if X[i] is None:
                  # if channel is None, fill it with zeros of the correct
                  # dtype
                  dt = out_array.dtype
                  if output_pixel_vals:
                      dt = 'uint8'
                  out_array[:, :, i] = np.zeros(out_shape,
                          dtype=dt) + channel_defaults[i]
              else:
                  # use a recurrent call to compute the channel and store it
                  # in the output
                  out_array[:, :, i] = tile_raster_images(
                      X[i], img_shape, tile_shape, tile_spacing,
                      scale_rows_to_unit_interval, output_pixel_vals)
          return out_array

      else:
          # if we are dealing with only one channel
          H, W = img_shape
          Hs, Ws = tile_spacing

          # generate a matrix to store the output
          dt = X.dtype
          if output_pixel_vals:
              dt = 'uint8'
          out_array = np.zeros(out_shape, dtype=dt)

          for tile_row in xrange(tile_shape[0]):
              for tile_col in xrange(tile_shape[1]):
                  if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                      this_x = X[tile_row * tile_shape[1] + tile_col]
                      if scale_rows_to_unit_interval:
                          # if we should scale values to be between 0 and 1
                          # do this by calling the `scale_to_unit_interval`
                          # function
                          this_img = self.scale_to_unit_interval(
                              this_x.reshape(img_shape))
                      else:
                          this_img = this_x.reshape(img_shape)
                      # add the slice to the corresponding position in the
                      # output array
                      c = 1
                      if output_pixel_vals:
                          c = 255
                      out_array[
                          tile_row * (H + Hs): tile_row * (H + Hs) + H,
                          tile_col * (W + Ws): tile_col * (W + Ws) + W
                          ] = this_img * c
          return out_array


