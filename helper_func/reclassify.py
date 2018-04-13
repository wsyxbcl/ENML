# To reclassify and name data according to folders(named 0~#class) they belong

import os

dataset_dir = '/mnt/c/code/ENML/data/20180407_reclassified_1'
num_classes = 5
for c in range(num_classes):
    print('class '+str(c))
    for i, filename in enumerate(os.listdir(dataset_dir+'/'+str(c))):
        print('renaming '+filename)
        os.rename(dataset_dir+'/'+str(c)+'/'+filename, dataset_dir+'/'+str(c)+'/'+'training00'+str(c)+'_'+str(i+1)+'.csv')
