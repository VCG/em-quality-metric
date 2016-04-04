from train_cnn_viren import TrainCNN
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import pickle

import os
import uuid

from StringIO import StringIO
import sys

import argparse
import time

if __name__ == '__main__':


  parser = argparse.ArgumentParser()
  parser.add_argument("-r", "--runmode", type=str, help='local or cluster', default='local')
  #parser.add_argument("-d", "--datapath", type=str, help="the datapath", default='/Volumes/DATA1/EMQM_DATA/ac3x75/')
  parser.add_argument("-p", "--patchpath", type=str, help="the patch folder in the datapath", default='patches_4th_small')
  #parser.add_argument("-o", "--outputpath", type=str, help="the outputpath", default='/Volumes/DATA1/split_cnn/')
  parser.add_argument("-e", "--epochs", type=int, help="the number of epochs", default=5)
  parser.add_argument("-b", "--batchsize", type=int, help="the batchsize", default=100)
  parser.add_argument("-l", "--learning_rate", type=float, help="the learning rate", default=0.0001)
  parser.add_argument("-m", "--momentum", type=float, help="the momentum", default=0.9)
  parser.add_argument("-f1", "--filters1", type=int, help="the number of filters 1", default=16)
  parser.add_argument("-fs1", "--filtersize1", type=int, help="the filtersize 1", default=13)
  parser.add_argument("-f2", "--filters2", type=int, help="the number of filters 2", default=16)
  parser.add_argument("-fs2", "--filtersize2", type=int, help="the filtersize 2", default=13)
  parser.add_argument("-tcl", "--thirdconvlayer", type=str, help="use a third conv layer", default='False')
  parser.add_argument("-f3", "--filters3", type=int, help="the number of filters 2", default=16)
  parser.add_argument("-fs3", "--filtersize3", type=int, help="the filtersize 2", default=13)  
  parser.add_argument("-u", "--uuid", type=str, help='the uuid', default=str(uuid.uuid4()))
  parser.add_argument("-i", "--inputs", nargs='+', help='the list of input patches', default=['image', 'prob', 'merged_array'])
  parser.add_argument("-rp", "--rotate_patches", type=str, help="rotate patches after each epoch", default="True")
  parser.add_argument("-ps", "--patchsize", type=int, help="the patchsize (one dim)", default=75)

  args = parser.parse_args()



  if args.runmode == 'local':
    args.datapath = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
    args.outputpath = '/Volumes/DATA1/split_cnn/'+args.patchpath+'/'
  elif args.runmode == 'cluster':
    args.datapath = '/n/regal/pfister_lab/haehn/'
    args.outputpath = '/n/regal/pfister_lab/haehn/split_cnn/'+args.patchpath+'/'

  args_as_text = vars(args)

  if args.thirdconvlayer == 'False':
    args.thirdconvlayer = False
  else:
    args.thirdconvlayer = True

  if args.rotate_patches == 'False':
    args.rotate_patches = False
  else:
    args.rotate_patches = True

  print args_as_text



  #
  #
  #
  #
  UID = args.uuid
  OUTPUT_PATH = args.outputpath+UID
  if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

  with open(OUTPUT_PATH+os.sep+'configuration.txt', 'w') as f:
    f.write(str(args_as_text))


  def store_network(s, layers, epoch):

    print 'Storing network..'

    with open(OUTPUT_PATH+os.sep+'network_'+str(epoch)+'.p', 'wb') as f:
      pickle.dump(layers, f)



  def store_filters(s, layers, epoch):

    # grab learning pics
    for i in s._inputs:
      filters = Image.fromarray(s.visualize_filters(layers[i]['c2d_layer']))
      filters.save(OUTPUT_PATH+os.sep+i+'_filters_'+str(epoch)+'.png')  

    epochs = np.arange(0, len(s._training_loss))
    training_loss = s._training_loss
    validation_loss = s._validation_loss
    validation_acc = s._validation_acc

    # create plot
    fig, ax = plt.subplots()
    ax.plot(epochs, training_loss, 'k--', label='Training Loss')
    ax.plot(epochs, validation_loss, 'k:', label='Validation Loss')
    ax.plot(epochs, validation_acc, 'k', label='Validation Accuracy')
    ax.set_yscale('log')

    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper center', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    plt.savefig(OUTPUT_PATH+os.sep+'graph_'+str(epoch)+'.png')

    store_network(s, layers, epoch)


  s = TrainCNN()
  s._DATA_PATH = args.datapath
  s._PATCH_PATH = s._DATA_PATH+args.patchpath+'/'
  s._EPOCHS = args.epochs
  s._BATCH_SIZE = args.batchsize
  s._LEARNING_RATE = args.learning_rate
  s._MOMENTUM = args.momentum
  s._INPUT_SHAPE = (None, 1, 75, 75)
  s._NO_FILTERS = args.filters1
  s._FILTER_SIZE = (args.filtersize1,args.filtersize1)
  s._NO_FILTERS2 = args.filters2
  s._FILTER_SIZE2 = (args.filtersize2,args.filtersize2)
  s._THIRD_CONV_LAYER = args.thirdconvlayer
  s._NO_FILTERS3 = args.filters3
  s._FILTER_SIZE3 = (args.filtersize3,args.filtersize3)
  s._EPOCH_CALLBACK = store_filters
  s._CONV_CALLBACK = store_network
  s._inputs = args.inputs
  s._rotate_patches = args.rotate_patches

  s._patch_size = (args.patchsize, args.patchsize)
  s._INPUT_SHAPE = (None, 1, s._patch_size[0], s._patch_size[1])

  print 'Using the following inputs:', s._inputs
  print 'Network configured.. running now!'
  # old_stdout = sys.stdout
  # old_stderr = sys.stderr
  # stdout_buffer = StringIO()
  # stderr_buffer = StringIO()
  # sys.stdout = stdout_buffer
  # sys.stderr = stderr_buffer


  t0 = time.time()

  #
  # RUN THE NETWORK
  #
  layers = s.run()

  t1 = time.time()
  total_time = t1 - t0


  # grab final values
  test_loss = s._test_loss[0]
  test_acc = s._test_acc[0]

  #
  # STORE CONSOLE OUTPUT
  #
  with open(OUTPUT_PATH+os.sep+'final_test_loss_'+str(test_loss)+'___test_acc_'+str(test_acc)+'.txt', 'w') as f:
    f.write('pretty good?')

  # with open(OUTPUT_PATH+os.sep+'out.log', 'w') as f:
  #   f.write(stdout_buffer.getvalue())

  # with open(OUTPUT_PATH+os.sep+'err.log', 'w') as f:
  #   f.write(stderr_buffer.getvalue())

  with open(OUTPUT_PATH+os.sep+'time.log', 'w') as f:
    f.write(str(total_time))

  # sys.stdout = old_stdout
  # sys.stderr = old_stderr

  print 'Stored in', OUTPUT_PATH
  print 'All done. Ciao!'
