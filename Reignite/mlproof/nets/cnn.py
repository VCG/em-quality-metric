import cPickle as pickle

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from lasagne.updates import nesterov_momentum

from helper import *

class CNN(object):

    def __init__(self, *args, **kwargs):
        '''
        '''

        kwargs['update'] = nesterov_momentum
        kwargs['update_learning_rate'] = 0.001
        kwargs['update_momentum'] = 0.9
        # # update_learning_rate=theano.shared(float32(0.03)),
        # # update_momentum=theano.shared(float32(0.9)),

        kwargs['regression'] = False
        kwargs['batch_iterator_train'] = MyBatchIterator(batch_size=100)
        kwargs['batch_iterator_test'] = MyBatchIterator(batch_size=100)
        kwargs['max_epochs'] = 500
        kwargs['train_split'] = TrainSplit(eval_size=0.25)
        kwargs['on_epoch_finished'] = [
                # AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                # AdjustVariable('update_momentum', start=0.9, stop=0.999),
                EarlyStopping(patience=50),
            ]
        
        kwargs['verbose'] = True

        print 'CNN configuration:', self.__doc__

        cnn = NeuralNet(*args, **kwargs)
        self.__class__ = cnn.__class__
        self.__dict__ = cnn.__dict__        

    def store_values(self, filename):
        '''
        '''
        v = self.get_all_params_values()
        with open(filename, 'wb') as f:
            pickle.dump(v, f)

        print 'Stored everything.'

    def load_values(self, filename):
        '''
        '''
        with open(filename, 'rb') as f:
            v = pickle.load(f)

        self.load_params_from(v)

        print 'Loaded everything.'

