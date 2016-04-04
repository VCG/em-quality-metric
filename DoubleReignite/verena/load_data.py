import urllib
dataset='mnist.pkl.gz'
origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
print 'Downloading data from %s' % origin
urllib.urlretrieve(origin, dataset)
