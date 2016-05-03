print '***** MLProof *****'

from theano.sandbox.cuda import dnn

print 'CuDNN support:', dnn.dnn_available()
