import glob

files = glob.glob('/n/home05/haehn/slurm/out-*.txt')
for f in files:

  with open(f, 'r') as l:
    lines = l.readlines()
    print lines[-5]
    