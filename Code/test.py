import os
import time
import _metrics

DATA_PATH = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
GOLD_PATH = os.path.join(DATA_PATH,'gold/')
IMAGE_PATH = os.path.join(DATA_PATH,'input/')
TRAINING_PATH = os.path.join(DATA_PATH,'training/')


gold = _metrics.Util.read(GOLD_PATH+'*.tif')
images = _metrics.Util.read(IMAGE_PATH+'*.tif')

t0 = time.time()
splits = 0
data = []
#for s in _metrics.SplitError.generate(images[0:1,600:800,200:400], gold[0:1,600:800,200:400], 1, thumb=False, rotate=True):
for s in _metrics.SplitError.generate(images[0:2], gold[0:2], 10, thumb=False, rotate=True):    
    splits += 1
    #data.append(s)

t1 = time.time()

total = t1-t0

print total, 'seconds for', splits, 'splits'
