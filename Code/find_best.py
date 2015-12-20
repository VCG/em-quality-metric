import argparse
import glob
import os


if __name__ == '__main__':


  parser = argparse.ArgumentParser()
  # parser.add_argument("-r", "--runmode", type=str, help='local or cluster', default='local')
  #parser.add_argument("-d", "--datapath", type=str, help="the datapath", default='/Volumes/DATA1/EMQM_DATA/ac3x75/')
  parser.add_argument("-p", "--patchpath", type=str, help="the patch folder in the datapath", default='patches_small')
  parser.add_argument("-s", "--sort", type=str, help="sort by loss or accuracy", default='acc')
  parser.add_argument("-n", "--number", type=int, help="show only N entries", default=-1)
  parser.add_argument("-v", "--verbose", type=bool, help='show configuration per entry', default=False)


  args = parser.parse_args()

  # if args.runmode == 'local':
  # args.datapath = '/Volumes/DATA1/EMQM_DATA/ac3x75/'
  args.outputpath = '/Volumes/DATA1/split_cnn/'+args.patchpath+'/'
  # elif args.runmode == 'cluster':
  #   args.datapath = '/n/regal/pfister_lab/haehn/'
  #   args.outputpath = '/n/regal/pfister_lab/haehn/split_cnn/'+args.patchpath+'/'

  args_as_text = vars(args)

  print args_as_text  

  losses = []
  accs = []
  ids = []

  folders = glob.glob(args.outputpath + '*')
  for f in folders:

    if not os.path.isdir(f):
      continue

    final_test = glob.glob(f+os.sep+'final_test*')

    if len(final_test) > 0:

      final_test_file = os.path.basename(final_test[0])
      final_test_file_splitted = final_test_file.split('final_test_loss_')[1].split('__')

      loss = final_test_file_splitted[0]
      acc = final_test_file_splitted[1].split('_test_acc_')[1].split('.txt')[0]



      if loss != 'nan':
        # print loss, acc
        accs.append(float(acc))
        losses.append(float(loss))
        ids.append(os.path.basename(f))

        # print accs, losses, ids

  # sorted_list = [z,y,x for (z,y,x) in sorted(zip(accs, losses, ids))]

  results = zip(accs, losses, ids)

  key = 0
  if args.sort == 'loss':
    key = 1

  sorted_results = sorted(results, key=lambda tup: tup[key], reverse=True)
  # print accs

  # print sorted_list

  if args.number == -1:
    args.number = len(sorted_results)

  print ('ACC', 'LOSS', 'ID')

  for s in sorted_results[0:args.number]:
    print s
  
    if args.verbose:
      # grab configuration
      with open(args.outputpath+s[2]+os.sep+'configuration.txt', 'r') as f:
        print f.readlines()
      print '-'*80
