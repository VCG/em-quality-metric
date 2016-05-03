import cPickle as pickle

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from lasagne.updates import nesterov_momentum
import theano

from helper import *

class CNN(object):

    def __init__(self, *args, **kwargs):
        '''
        '''

        kwargs['update'] = nesterov_momentum
        kwargs['update_learning_rate'] = theano.shared(float32(0.03))#0.001
        kwargs['update_momentum'] = theano.shared(float32(0.9))#0.9
        # update_learning_rate=theano.shared(float32(0.03)),
        # update_momentum=theano.shared(float32(0.9)),

        kwargs['regression'] = False
        kwargs['batch_iterator_train'] = MyBatchIterator(batch_size=300)
        kwargs['batch_iterator_test'] = MyBatchIterator(batch_size=300)
        kwargs['max_epochs'] = 1000
        kwargs['train_split'] = TrainSplit(eval_size=0.25)
        kwargs['on_epoch_finished'] = [
                AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                AdjustVariable('update_momentum', start=0.9, stop=0.999),
                EarlyStopping(patience=30),
            ]
        
        kwargs['verbose'] = True

        print 'CNN configuration:', self.__doc__

        cnn = NeuralNet(*args, **kwargs)
        self.__class__ = cnn.__class__
        self.__dict__ = cnn.__dict__        
