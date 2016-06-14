from lasagne import layers
from lasagne import nonlinearities

from cnn import CNN

class RGBANet(CNN):
    '''
    Our CNN with image, prob, merged_array, border_overlap as RGBA.
    '''

    def __init__(self):

        CNN.__init__(self,

            layers=[
                ('input', layers.InputLayer),

                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),

                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),

                ('hidden3', layers.DenseLayer),
                ('dropout3', layers.DropoutLayer),
                ('output', layers.DenseLayer),
            ],

            # input
            input_shape=(None, 4, 75, 75),

            # conv2d + pool + dropout
            conv1_filter_size=(13,13), conv1_num_filters=16,
            pool1_pool_size=(2,2),

            # conv2d + pool + dropout
            conv2_filter_size=(13,13), conv2_num_filters=16,
            pool2_pool_size=(2,2),

            # dense layer 1
            hidden3_num_units=256,
            hidden3_nonlinearity=nonlinearities.rectify,
            dropout3_p=0.5,

            # dense layer 2
            output_num_units=2,
            output_nonlinearity=nonlinearities.softmax

        )
