# Used to compare data collected from 2 different stations.

from dir_walker import walker
import re
import csv
import matplotlib.pyplot as plt
import random

plt.style.use('ggplot')

rootdir = 'C:/code/ENML/data/20171228/class3'

subdir1 = list(walker(rootdir, re.compile('(.*?).csv')))[0][1]
label1 = subdir1[(len(rootdir) - len(subdir1)):]
subdir2 = list(walker(rootdir, re.compile('(.*?).csv')))[-1][1]
label2 = subdir2[(len(rootdir) - len(subdir2)):]

x_range = range(7000, 7101)
fig = plt.figure()
i, j = (1, 1)
num_file = 0

for filename, subdir in walker(subdir1, re.compile('(.*?).csv')):
    if random.random() < 0.8:
        continue
    print(subdir+'/'+filename)
    num_file = num_file + 1
    print('#file: %d'%num_file)
    with open(subdir+'/'+filename, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
        x = [float(data_list[i][1]) for i in x_range]
        coordinate = [float(data_list[i][0]) for i in x_range]
        # Plot the data
        ax = fig.add_subplot(1, 1, 1)
        if i:
            ax.plot(coordinate, x, color='red', label=label1)
            i = 0
        else:
            ax.plot(coordinate, x, color='red', label='')

for filename, subdir in walker(subdir2, re.compile('(.*?).csv')):
    print(subdir+'/'+filename)
    num_file = num_file + 1
    print('#file: %d'%num_file)
    with open(subdir+'/'+filename, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)[1:]
        x = [float(data_list[i][1]) for i in x_range]
        coordinate = [float(data_list[i][0]) for i in x_range]
        # Plot the data
        ax = fig.add_subplot(1, 1, 1)
        if j:
            ax.plot(coordinate, x, color='black', label=label2)
            j = 0
        else:
            ax.plot(coordinate, x, color='black', label='')

ax.set_xlabel('time/ms')
ax.set_ylabel('current/A')
ax.legend(loc='best')
plt.savefig(rootdir+'/energylab.png', dpi=300)