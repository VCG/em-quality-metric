from split_cnn import SplitCNN
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import os
import uuid

from StringIO import StringIO
import sys



#
#
#
#
UID = str(uuid.uuid4())
OUTPUT_PATH = '/Volumes/DATA1/split_cnn/'+UID
os.makedirs(OUTPUT_PATH)


def store_filters(s, layers, epoch):

  # grab learning pics
  image_filters = Image.fromarray(s.visualize_filters(layers['image']['c2d_layer']))
  image_filters.save(OUTPUT_PATH+os.sep+'image_filters_'+str(epoch)+'.png')
  prob_filters = Image.fromarray(s.visualize_filters(layers['prob']['c2d_layer']))
  prob_filters.save(OUTPUT_PATH+os.sep+'prob_filters_'+str(epoch)+'.png')
  binary1_filters = Image.fromarray(s.visualize_filters(layers['binary1']['c2d_layer']))
  binary1_filters.save(OUTPUT_PATH+os.sep+'binary1_filters_'+str(epoch)+'.png')
  binary2_filters = Image.fromarray(s.visualize_filters(layers['binary2']['c2d_layer']))
  binary2_filters.save(OUTPUT_PATH+os.sep+'binary2_filters_'+str(epoch)+'.png')
  overlap_filters = Image.fromarray(s.visualize_filters(layers['overlap']['c2d_layer']))
  overlap_filters.save(OUTPUT_PATH+os.sep+'overlap_filters_'+str(epoch)+'.png')  

  epochs = np.arange(0, len(s._training_loss))
  training_loss = s._training_loss
  validation_loss = s._validation_loss
  validation_acc = s._validation_acc

  # create plot
  fig, ax = plt.subplots()
  ax.plot(epochs, training_loss, 'k--', label='Training Loss')
  ax.plot(epochs, validation_loss, 'k:', label='Validation Loss')
  ax.plot(epochs, validation_acc, 'k', label='Validation Accuracy')

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


s = SplitCNN()
s._DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
s._PATCH_PATH = s._DATA_PATH+'patches_small/'
s._EPOCHS = 5
s._BATCH_SIZE = 5
s._LEARNING_RATE = 0.0001
s._MOMENTUM = 0.9
s._INPUT_SHAPE = (None, 1, 75, 75)
s._NO_FILTERS = 32
s._FILTER_SIZE = (5,5)
s._NO_FILTERS2 = 32
s._FILTER_SIZE2 = (5,5)
s._EPOCH_CALLBACK = store_filters

print 'Network configured.. running now!'
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_buffer = StringIO()
stderr_buffer = StringIO()
sys.stdout = stdout_buffer
sys.stderr = stderr_buffer


#
# RUN THE NETWORK
#
layers = s.run()





# grab final values
test_loss = s._test_loss[0]
test_acc = s._test_acc[0]

#
# STORE CONSOLE OUTPUT
#
with open(OUTPUT_PATH+os.sep+'final_test_loss_'+str(test_loss)+'___test_acc_'+str(test_acc)+'.txt', 'w') as f:
  f.write('pretty good?')

with open(OUTPUT_PATH+os.sep+'out.log', 'w') as f:
  f.write(stdout_buffer.getvalue())

with open(OUTPUT_PATH+os.sep+'err.log', 'w') as f:
  f.write(stderr_buffer.getvalue())

sys.stdout = old_stdout
sys.stderr = old_stderr

print 'All done. Ciao!'
