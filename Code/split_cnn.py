import numpy as np
import os
import sys
import theano
import theano.tensor as T
import time

import lasagne



class SplitCNN(object):

  def __init__(self):
    '''
    '''
    #self.initialize()
    self._dbg = None

    self._DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
    self._PATCH_PATH = os.path.join(self._DATA_PATH,'patches_small/')

    self._EPOCHS = 5
    self._BATCH_SIZE = 5#00
    self._LEARNING_RATE = 0.0001
    self._MOMENTUM = 0.9
    self._INPUT_SHAPE = (None, 1, 75, 75)#(None, 75, 75)#(None, 1, 75, 75)
    self._NO_FILTERS = 32
    self._FILTER_SIZE = (5,5)
    self._NO_FILTERS2 = 32
    self._FILTER_SIZE2 = (5,5)
    self._THIRD_CONV_LAYER = False
    self._NO_FILTERS3 = 32
    self._FILTER_SIZE3 = (5,5)    
    self._EPOCH_CALLBACK = None

    self._training_loss = []
    self._validation_loss = []
    self._validation_acc = []
    self._test_loss = []
    self._test_acc = []


  # ################## Download and prepare the MNIST dataset ##################
  # This is just some way of getting the MNIST dataset from an online location
  # and loading it into numpy arrays. It doesn't involve Lasagne at all.

  def load_dataset(self):
      # # We first define a download function, supporting both Python 2 and 3.
      # if sys.version_info[0] == 2:
      #     from urllib import urlretrieve
      # else:
      #     from urllib.request import urlretrieve

      # def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
      #     print("Downloading %s" % filename)
      #     urlretrieve(source + filename, filename)

      # # We then define functions for loading MNIST images and labels.
      # # For convenience, they also download the requested files if needed.
      # import gzip

      # def load_mnist_images(filename):
      #     if not os.path.exists(filename):
      #         download(filename)
      #     # Read the inputs in Yann LeCun's binary format.
      #     with gzip.open(filename, 'rb') as f:
      #         data = np.frombuffer(f.read(), np.uint8, offset=16)
      #     # The inputs are vectors now, we reshape them to monochrome 2D images,
      #     # following the shape convention: (examples, channels, rows, columns)
      #     data = data.reshape(-1, 1, 28, 28)
          
      #     # The inputs come as bytes, we convert them to float32 in range [0,1].
      #     # (Actually to range [0, 255/256], for compatibility to the version
      #     # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
      #     return data / np.float32(256)

      # def load_mnist_labels(filename):
      #     if not os.path.exists(filename):
      #         download(filename)
      #     # Read the labels in Yann LeCun's binary format.
      #     with gzip.open(filename, 'rb') as f:
      #         data = np.frombuffer(f.read(), np.uint8, offset=8)
      #     # The labels are vectors of integers now, that's exactly what we want.
      #     return data

      # # We can now download and read the training and test set images and labels.
      # X_train = load_mnist_images('train-images-idx3-ubyte.gz')
      # y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
      # X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
      # y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

      # # We reserve the last 10000 training examples for validation.
      # X_train, X_val = X_train[:-10000], X_train[-10000:]
      # y_train, y_val = y_train[:-10000], y_train[-10000:]

      # # We just return all the arrays in order, as expected in main().
      # # (It doesn't matter how we do this as long as we can read them again.)
      # return X_train, y_train, X_val, y_val, X_test, y_test    

      #
      #
      # LOAD PATCHES
      #
      #
      training = np.load(self._PATCH_PATH+'train.npz')
      training_targets = np.load(self._PATCH_PATH+'train_targets.npz')

      #
      # we also normalize all binary images as uint8
      #
      training = {
        'image': training['image'].reshape(-1, 1, 75, 75),
        'prob': training['prob'].reshape(-1, 1, 75, 75),
        'binary1': training['binary1'].reshape(-1, 1, 75, 75)*255,
        'binary2': training['binary2'].reshape(-1, 1, 75, 75)*255,
        'overlap': training['overlap'].reshape(-1, 1, 75, 75)*255, 
      }

      training_targets = training_targets['targets'].astype(np.uint8)



      val = np.load(self._PATCH_PATH+'val.npz')
      val_targets = np.load(self._PATCH_PATH+'val_targets.npz')

      #
      # we also normalize all binary images as uint8
      #
      val = {
        'image': val['image'].reshape(-1, 1, 75, 75),
        'prob': val['prob'].reshape(-1, 1, 75, 75),
        'binary1': val['binary1'].reshape(-1, 1, 75, 75)*255,
        'binary2': val['binary2'].reshape(-1, 1, 75, 75)*255,
        'overlap': val['overlap'].reshape(-1, 1, 75, 75)*255, 
      }

      val_targets = val_targets['targets'].astype(np.uint8)



      test = np.load(self._PATCH_PATH+'test.npz')
      test_targets = np.load(self._PATCH_PATH+'test_targets.npz')

      #
      # we also normalize all binary images as uint8
      #
      test = {
        'image': test['image'].reshape(-1, 1, 75, 75),
        'prob': test['prob'].reshape(-1, 1, 75, 75),
        'binary1': test['binary1'].reshape(-1, 1, 75, 75)*255,
        'binary2': test['binary2'].reshape(-1, 1, 75, 75)*255,
        'overlap': test['overlap'].reshape(-1, 1, 75, 75)*255, 
      }

      test_targets = test_targets['targets'].astype(np.uint8)

      #val = np.load(PATCH_PATH+'val.npz')
      #val_targets = np.load(PATCH_PATH+'val_targets.npz')

      return training, training_targets, val, val_targets, test, test_targets



  def run(self, num_epochs=5):
    '''
    '''
    # Load the dataset
    print("Loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset()
    # X_val = X_test = X_train
    # y_val = y_test = y_train

    # Prepare Theano variables for inputs and targets
    image_var = T.tensor4('image')
    prob_var = T.tensor4('prob')
    binary1_var = T.tensor4('binary1')
    binary2_var = T.tensor4('binary2')
    overlap_var = T.tensor4('overlap')
    target_var = T.ivector('targets')

    layers = self.build(image_var, prob_var, binary1_var, binary2_var, overlap_var, self._THIRD_CONV_LAYER)
    network = layers['dense']['network']

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=self._LEARNING_RATE, momentum=self._MOMENTUM)

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
    train_fn = theano.function([image_var, prob_var, binary1_var, binary2_var, overlap_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([image_var, prob_var, binary1_var, binary2_var, overlap_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:

    for epoch in range(self._EPOCHS):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in self.iterate_minibatches(X_train, y_train, self._BATCH_SIZE, shuffle=False):
            images, probs, binary1s, binary2s, overlaps, targets = batch
            self._dbg = images, probs, binary1s
            train_err += train_fn(images, probs, binary1s, binary2s, overlaps, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in self.iterate_minibatches(X_val, y_val, self._BATCH_SIZE, shuffle=False):
            #inputs, targets = batch
            images, probs, binary1s, binary2s, overlaps, targets = batch
            err, acc = val_fn(images, probs, binary1s, binary2s, overlaps, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, self._EPOCHS, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        self._training_loss.append(train_err / train_batches)
        self._validation_loss.append(val_err / val_batches)
        self._validation_acc.append(val_acc / val_batches * 100)

        # call the epoch callback
        if epoch % (self._EPOCHS / 5) == 0:
          self._EPOCH_CALLBACK(self, layers, epoch)

    # one final callback
    self._EPOCH_CALLBACK(self, layers, epoch)

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in self.iterate_minibatches(X_test, y_test, self._BATCH_SIZE, shuffle=False):
        # inputs, targets = batch
        # err, acc = val_fn(inputs, inputs, inputs, inputs, inputs, targets)
        images, probs, binary1s, binary2s, overlaps, targets = batch
        err, acc = val_fn(images, probs, binary1s, binary2s, overlaps, targets)        
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    self._test_loss.append(test_err / test_batches)
    self._test_acc.append(test_acc / test_batches * 100)

    return layers


  def gen_network(self, layers, name, input_var, third_conv_layer=False):
    '''
    '''

    layers[name] = {}

    # Input layer, as usual:
    input_layer = lasagne.layers.InputLayer(shape=self._INPUT_SHAPE,
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    layers[name]['input_layer'] = input_layer

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    c2d_layer = lasagne.layers.Conv2DLayer(
            input_layer, num_filters=self._NO_FILTERS, filter_size=self._FILTER_SIZE,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    layers[name]['c2d_layer'] = c2d_layer

    # Max-pooling layer of factor 2 in both dimensions:
    max_pool_layer = lasagne.layers.MaxPool2DLayer(c2d_layer, pool_size=(2, 2))

    layers[name]['max_pool_layer'] = max_pool_layer

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    c2d_layer2 = lasagne.layers.Conv2DLayer(
            max_pool_layer, num_filters=self._NO_FILTERS2, filter_size=self._FILTER_SIZE2,
            nonlinearity=lasagne.nonlinearities.rectify)

    layers[name]['c2d_layer2'] = c2d_layer2

    max_pool_layer2 = lasagne.layers.MaxPool2DLayer(c2d_layer2, pool_size=(2, 2))

    layers[name]['max_pool_layer2'] = max_pool_layer2

    layers[name]['network'] = max_pool_layer2

    if third_conv_layer:

      # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
      c2d_layer3 = lasagne.layers.Conv2DLayer(
              max_pool_layer, num_filters=self._NO_FILTERS3, filter_size=self._FILTER_SIZE3,
              nonlinearity=lasagne.nonlinearities.rectify)

      layers[name]['c2d_layer3'] = c2d_layer3

      max_pool_layer3 = lasagne.layers.MaxPool2DLayer(c2d_layer3, pool_size=(2, 2))

      layers[name]['max_pool_layer3'] = max_pool_layer3

      layers[name]['network'] = max_pool_layer3

    


    return layers





  def build(self, input_image, input_prob, input_binary1, input_binary2, input_overlap, third_conv_layer=False):
    '''
    '''
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    layers = {}

    layers = self.gen_network(layers, 'image', input_image, third_conv_layer)
    layers = self.gen_network(layers, 'prob', input_prob, third_conv_layer)
    layers = self.gen_network(layers, 'binary1', input_binary1, third_conv_layer)
    layers = self.gen_network(layers, 'binary2', input_binary2, third_conv_layer)
    layers = self.gen_network(layers, 'overlap', input_overlap, third_conv_layer)

    #
    #
    # MERGE LAYER
    #
    #
    layers['merged'] = {}

    merged = lasagne.layers.ConcatLayer([layers['image']['network'], layers['prob']['network'], layers['binary1']['network'], layers['binary2']['network'], layers['overlap']['network']])

    layers['merged']['network'] = merged


    #
    # DENSE LAYERS
    #

    layers['dense'] = {}

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    dense_layer1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(merged, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    layers['dense']['dense_layer1'] = dense_layer1

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    dense_layer2 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(dense_layer1, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    layers['dense']['dense_layer2'] = dense_layer2

    layers['dense']['network'] = dense_layer2

    return layers
























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
      assert len(inputs['binary1']) == len(targets)
      assert len(inputs['binary2']) == len(targets)
      assert len(inputs['overlap']) == len(targets)
      if shuffle:
          indices = np.arange(len(inputs))
          np.random.shuffle(indices)
      for start_idx in range(0, len(inputs['image']) - batchsize + 1, batchsize):
          if shuffle:
              excerpt = indices[start_idx:start_idx + batchsize]
          else:
              excerpt = slice(start_idx, start_idx + batchsize)
          yield inputs['image'][excerpt], inputs['prob'][excerpt], inputs['binary1'][excerpt], inputs['binary2'][excerpt], inputs['overlap'][excerpt], targets[excerpt]

  def visualize_filters(self, layer):

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


