import glob
import os
import time

files = glob.glob('/n/home05/haehn/slurm/out-*_low_lr.txt')
for f in files:

  mtime = os.path.getmtime(f)

  if time.time()-mtime > 10000:
    continue
  print f
  with open(f, 'r') as l:
    lines = l.readlines()
    #print lines[-5:]
    for g in lines[-8:]:
      print g.strip('\n')
    print '-'*80
    
