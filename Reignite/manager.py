import cPickle as pickle
import os

from util import Util
from uitools import UITools

class Manager(object):

  def __init__(self):
    '''
    '''
    self._data_path = '/home/d/dojo_xp/data/'
    self._merge_errors = None

  def start( self ):
    '''
    '''
    # load data
    input_image, input_prob, input_gold, input_rhoana, dojo_bbox = Util.read_dojo_data() 
    self._input_image = input_image
    self._input_prob = input_prob
    self._input_gold = input_gold
    self._input_rhoana = input_rhoana
    self._dojo_bbox = dojo_bbox

    self._merge_errors = self.load_merge_errors()
    self._bigM = self.load_split_errors()

    self._cnn = UITools.load_cnn()

    print 'VI at start:', UITools.VI(self._input_gold, self._input_rhoana)[0]

  def load_merge_errors(self):
    '''
    '''
    with open(os.path.join(self._data_path, 'merges2.p'), 'rb') as f:
      merge_errors = pickle.load(f)

    return sorted(merge_errors, key=lambda x: x[3], reverse=False)

  def load_split_errors(self):
    '''
    '''
    with open(os.path.join(self._data_path, 'bigM.p'), 'rb') as f:
      bigM = pickle.load(f)

    return bigM

  def get_next_merge_error(self):
    '''
    '''
    return self._merge_errors[0]

  def get_next_split_error(self):
    '''
    '''
    z, labels, prediction = UITools.find_next_split_error(self._bigM)

    self._split_error = (z, labels, prediction)

    return self._split_error

  def get_merge_error_image(self, merge_error, number):

    border = merge_error[3][number][1]


    z = merge_error[0]
    label = merge_error[1]

    input_image = self._input_image
    input_prob = self._input_prob
    input_rhoana = self._input_rhoana


    a,b,c,d,e,f,g,h,i = UITools.get_merge_error_image(input_image[z], input_rhoana[z], label, border)

    border_before = b
    labels_before = h
    border_after = c
    labels_after = i
    slice_overview = g

    return border_before, border_after, labels_before, labels_after, slice_overview
    
  def get_split_error_image(self, split_error, number=1):

    z = split_error[0]
    labels = split_error[1]

    input_image = self._input_image
    input_prob = self._input_prob
    input_rhoana = self._input_rhoana

    a,b,c,d,e,f = UITools.get_split_error_image(input_image[z], input_rhoana[z], labels)

    labels_before = b
    borders_before = c
    borders_after = d
    labels_after = e
    slice_overview = f

    return borders_before, borders_after, labels_before, labels_after, slice_overview

  def correct_merge(self, clicked_correction):

    input_image = self._input_image
    input_prob = self._input_prob
    input_rhoana = self._input_rhoana

    if not clicked_correction == 'current':
        clicked_correction = int(clicked_correction)

        #
        # correct the merge
        #
        merge_error = self._merge_errors[0]
        number = clicked_correction
        border = merge_error[3][number][1]
        z = merge_error[0]
        label = merge_error[1]

        a,b,c,d,e,f,g,h,i = UITools.get_merge_error_image(input_image[z], input_rhoana[z], label, border)

        new_rhoana = f

        input_rhoana[z] = new_rhoana

        #
        # and remove the original label from our bigM matrix
        #
        self._bigM[z][label,:] = -3
        self._bigM[z][:,label] = -3
        # TODO a better way would be to re-calculate bigM but this takes too much time on the fly

        # self.finish()

    # remove merge error
    del self._merge_errors[0]

    mode = 'merge'
    if len(self._merge_errors) == 0:

        mode = 'split'

    return mode

  def correct_split(self, clicked_correction):

    input_image = self._input_image
    input_prob = self._input_prob
    input_rhoana = self._input_rhoana

    split_error = self._split_error

    z = split_error[0]
    labels = split_error[1]
    m = self._bigM[z]

    if clicked_correction == 'current':
        # we skip this split
        new_m = UITools.skip_split(m, labels[0], labels[1])
        self._bigM[z] = new_m

    else:
        # we correct this split
        new_m, new_rhoana = UITools.correct_split(self._cnn, m, input_image[z], input_prob[z], input_rhoana[z], labels[0], labels[1], oversampling=False)
        self._bigM[z] = new_m
        self._input_rhoana[z] = new_rhoana

        # self.finish()
        
    return 'split'


  def finish(self):

    vi = UITools.VI(self._input_gold, self._input_rhoana)
    print 'New VI', vi[0]

    # store our changed rhoana
    with open(os.path.join(self._data_path, 'ui_results.p'), 'wb') as f:
        pickle.dump(self._input_rhoana, f)

    print 'All stored.'








