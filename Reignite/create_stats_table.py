import sys
import os
from stats import Stats
from string import Template
import cPickle as pickle

if __name__ == '__main__':

  row_template = Template('''
<tr>
<td>$CNN_NAME</td>
<td>$TRAINING</td>
<td>$CONFIGURATION</td>
<td>$TEST_LOSS</td>
<td>$TEST_ACC</td>
<td>$SPLITS_GT_MEAN_VI</td>
<td>$SPLITS_GT_MEAN_VI_DIFF</td>
<td>$SPLITS_GT_MEAN_SURENESS</td>
<td>$SPLITS_RHOANA_MEAN_VI</td>
<td>$SPLITS_RHOANA_MEAN_VI_DIFF</td>
<td>$SPLITS_RHOANA_MEAN_SURENESS</td>
<td>$MERGES_GT_TOP1</td>
<td>$MERGES_GT_TOP2</td>
<td>$MERGES_GT_TOP3</td>
<td>$MERGES_GT_TOP4</td>
<td>$MERGES_GT_TOP5</td>
<td>$MERGES_RHOANA_TOP1</td>
<td>$MERGES_RHOANA_TOP2</td>
<td>$MERGES_RHOANA_TOP3</td>
<td>$MERGES_RHOANA_TOP4</td>
<td>$MERGES_RHOANA_TOP5</td>
</tr>
''')




  OUTPUT_PATH = '/Volumes/DATA1/cnn_analysis/'
  subdirs = os.listdir(OUTPUT_PATH)

  table = ''

  for d in subdirs:

    if d.startswith('.'):
      continue

    if d == 'gfx':
      continue

    if not os.path.isdir(OUTPUT_PATH+os.sep+d):
      continue

    with open(OUTPUT_PATH+os.sep+d+os.sep+'values.p', 'rb') as f:
        v = pickle.load(f)

    training = 'Rhoana'
    if v['TRAINING'].find('ground truth') != -1:
      training = 'Groundtruth'

    row = row_template.substitute(CNN_NAME="<a href='"+v['CNN_NAME']+"/index.html' target=_blank>"+v['CNN_NAME']+"</a>",
                                  TRAINING=training,
                                  CONFIGURATION=v['PATCHES'].replace('100','15').replace('src="', 'src="'+v['CNN_NAME']+'/'),
                                  TEST_LOSS=round(float(v['TEST_LOSS']),3),
                                  TEST_ACC=round(float(v['TEST_ACC']),3),
                                  SPLITS_GT_MEAN_VI=round(float(v['SPLITS_GT_MEAN_VI']),3),
                                  SPLITS_GT_MEAN_VI_DIFF=round(float(v['SPLITS_GT_MEAN_VI_DIFF']),3),
                                  SPLITS_GT_MEAN_SURENESS=round(float(v['SPLITS_GT_MEAN_SURENESS']),3),
                                  SPLITS_RHOANA_MEAN_VI=round(float(v['SPLITS_RHOANA_MEAN_VI']),3),
                                  SPLITS_RHOANA_MEAN_VI_DIFF=round(float(v['SPLITS_RHOANA_MEAN_VI_DIFF']),3),
                                  SPLITS_RHOANA_MEAN_SURENESS=round(float(v['SPLITS_RHOANA_MEAN_SURENESS']),3),
                                  MERGES_GT_TOP1=round(float(v['MERGES_GT_TOP1']),3),
                                  MERGES_GT_TOP2=round(float(v['MERGES_GT_TOP2']),3),
                                  MERGES_GT_TOP3=round(float(v['MERGES_GT_TOP3']),3),
                                  MERGES_GT_TOP4=round(float(v['MERGES_GT_TOP4']),3),
                                  MERGES_GT_TOP5=round(float(v['MERGES_GT_TOP5']),3),
                                  MERGES_RHOANA_TOP1=round(float(v['MERGES_RHOANA_TOP1']),3),
                                  MERGES_RHOANA_TOP2=round(float(v['MERGES_RHOANA_TOP2']),3),
                                  MERGES_RHOANA_TOP3=round(float(v['MERGES_RHOANA_TOP3']),3),
                                  MERGES_RHOANA_TOP4=round(float(v['MERGES_RHOANA_TOP4']),3),
                                  MERGES_RHOANA_TOP5=round(float(v['MERGES_RHOANA_TOP5']),3)
                                  )

    table += row+'\n'
    


with open(OUTPUT_PATH+'template_table.html','r') as f:
    t = Template(f.read())
    t_out = t.substitute(TABLE=table)

with open(OUTPUT_PATH+'index.html','w') as f:
    f.write(t_out)

print 'All done.'
