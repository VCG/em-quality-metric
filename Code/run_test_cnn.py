from split_cnn import SplitCNN
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

from test_cnn import TestCNN

if __name__ == '__main__':


  parser = argparse.ArgumentParser()
  parser.add_argument("-r", "--runmode", type=str, help='local or cluster', default='local')
  #parser.add_argument("-d", "--datapath", type=str, help="the datapath", default='/Volumes/DATA1/EMQM_DATA/ac3x75/')
  parser.add_argument("-p", "--patchpath", type=str, help="the patch folder in the datapath", default='patches_small')
  parser.add_argument("-i", "--id", type=str, help="the network id", default='')

  args = parser.parse_args()

  if args.runmode == 'local':
    args.datapath = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
    args.outputpath = '/Volumes/DATA1/split_cnn'
  elif args.runmode == 'cluster':
    args.datapath = '/n/regal/pfister_lab/haehn/'
    args.outputpath = '/n/regal/pfister_lab/haehn/split_cnn'

  args_as_text = vars(args)

  print args_as_text
  
  if args.id == '':
    print 'No network id'
    sys.exit(2)

  t = TestCNN(args.id)
  t._DATA_PATH = args.datapath
  t._PATCH = args.patchpath
  t._OUTPUT_PATH = args.outputpath

  layers = t.run()
