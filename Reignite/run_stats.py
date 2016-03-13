import sys
from stats import Stats

if __name__ == '__main__':

  trained_gt = True

  if int(sys.argv[1])==1:

    cnn_name = 'mine'
    cnn_patch_path = 'patches_4th'
    cnn_inputs = ['image', 'prob', 'binary', 'border_overlap']

  elif int(sys.argv[1])==2:  

    cnn_name = 'mine_large'
    cnn_patch_path = 'patches_4th'
    cnn_inputs = ['image', 'prob', 'binary', 'larger_border_overlap']

  elif int(sys.argv[1])==3:  

    cnn_name = 'viren_200'
    cnn_patch_path = 'patches_4th'
    cnn_inputs = ['image', 'prob', 'merged_array', 'dyn_obj', 'dyn_bnd']

  elif int(sys.argv[1])==4:  

    cnn_name = 'viren_overlap'
    cnn_patch_path = 'patches_4th'
    cnn_inputs = ['image', 'prob', 'merged_array', 'dyn_obj', 'dyn_bnd', 'border_overlap']

  elif int(sys.argv[1])==5:  

    cnn_name = 'viren_large'
    cnn_patch_path = 'patches_4th'
    cnn_inputs = ['image', 'prob', 'merged_array', 'dyn_obj', 'dyn_bnd', 'larger_border_overlap']

#
# Rhoana trained
#
  elif int(sys.argv[1])==6:  

    cnn_name = 'dcb5cb86-7e51-4a52-bb3f-bbb1195036fa'
    cnn_patch_path = 'patches_6th'
    cnn_inputs = ['image', 'prob', 'binary', 'border_overlap']
    trained_gt = False

  elif int(sys.argv[1])==7:  

    cnn_name = 'mine_merged_6_low_lr'
    cnn_patch_path = 'patches_6th'
    cnn_inputs = ['image', 'prob', 'merged_array', 'border_overlap']
    trained_gt = False

  elif int(sys.argv[1])==8:  

    cnn_name = 'viren_6_low_lr'
    cnn_patch_path = 'patches_6th'
    cnn_inputs = ['image', 'prob', 'merged_array', 'dyn_obj', 'dyn_bnd']
    trained_gt = False

  elif int(sys.argv[1])==9:  

    cnn_name = 'viren_large_6_low_lr'
    cnn_patch_path = 'patches_6th'
    cnn_inputs = ['image', 'prob', 'merged_array', 'dyn_obj', 'dyn_bnd', 'larger_border_overlap']
    trained_gt = False

  elif int(sys.argv[1])==10:

    cnn_name = 'viren_overlap_6_low_lr'
    cnn_patch_path = 'patches_6th'
    cnn_inputs = ['image', 'prob', 'merged_array', 'dyn_obj', 'dyn_bnd', 'border_overlap']
    trained_gt = False

  elif int(sys.argv[1])==11:

    cnn_name = 'mine_6_low_lr'
    cnn_patch_path = 'patches_6th'
    cnn_inputs = ['image', 'prob', 'binary', 'border_overlap']
    trained_gt = False

  elif int(sys.argv[1])==12:

    cnn_name = 'mine_large_6_low_lr'
    cnn_patch_path = 'patches_6th'
    cnn_inputs = ['image', 'prob', 'binary', 'larger_border_overlap']
    trained_gt = False

  elif int(sys.argv[1])==13:

    cnn_name = 'mine_merged_large_6_low_lr'
    cnn_patch_path = 'patches_6th'
    cnn_inputs = ['image', 'prob', 'merged_array', 'larger_border_overlap']
    trained_gt = False





  Stats.create(cnn_name, cnn_patch_path, cnn_inputs, trained_gt)

