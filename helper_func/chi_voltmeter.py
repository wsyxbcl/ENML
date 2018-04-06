# Using CHI electrochemical station as a voltmeter to collect V-t curve
# The program extracts an eight-second range of high-voltage period and calculate its standard deviation


from dir_walker import walker
import re
import matplotlib as plt
import numpy as np

rootdir = 'C:/code/ENML/test/chi_p'

def read_p(file, skiplines=4):
    """
    length: the length of required range of potential, 8 seconds as default
    skiplines: skip header
    """
    with open(file, 'rt') as f:
        for i in range(skiplines):
            next(f)
        time = []
        potential = []
        for line in f:
            time.append(float(line.split()[0]))
            potential.append(float(line.split()[1]))
    start_record = 0
    for i, p in enumerate(potential):
        if start_record and (sum(potential[i:i+50])/50 > 3e-3):
            time_last = time[i] - time[start_record]
        elif (sum(potential[i:i+50])/50 > 3e-3):
            start_record = i
        elif start_record:
            break
    print('time: '+str(time_last))
    print('std: '+str(np.std(potential[start_record+50:i])))
    return time[start_record+50:i], potential[start_record+50:i]

def write_p(file, time, potential):
    with open(file, 'w') as f:
        # f.write('time/ms,voltage/V')
        # f.write('\n')
        for i, p in enumerate(potential):
            f.write(str(time[i]))
            f.write(',')
            f.write(str(p))
            f.write('\n')

for filename, subdir in walker(rootdir, re.compile('(.*?).txt')):
    print(filename)
    time, potential = read_p(subdir+'/'+filename)
    write_p(subdir+'/'+filename[:-4]+'.csv', time, potential)