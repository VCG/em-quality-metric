from split_cnn import SplitCNN


#
#
#
#

s = SplitCNN()
s._DATA_PATH = '/n/home05/haehn/EMQM_DATA/ac3x75/'
s._PATCH_PATH = s._DATA_PATH+'patches_medium/'
s._EPOCHS = 50
s._BATCH_SIZE = 500
s._LEARNING_RATE = 0.0001
s._MOMENTUM = 0.9
s._INPUT_SHAPE = (None, 1, 75, 75)
s._NO_FILTERS = 32
s._FILTER_SIZE = (5,5)
s._NO_FILTERS2 = 32
s._FILTER_SIZE2 = (5,5)

layers = s.run()

