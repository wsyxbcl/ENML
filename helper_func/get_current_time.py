import os
import re
import csv
from dir_walker import walker

def get_i_t_ivium(source_file, output_file):
    """
    Get current and time information from the fucking weird ivium output
    ascii file and rewrite them to a csv file.
    """
    with open(source_file, 'rt') as f:
        a = []
        idx = [0, 2]
        for line in f:
            a.append([line.split()[i] for i in idx])
    with open(output_file, 'w') as f:
        for line in a:
            end = 0 # To judge the end in order to cancel the ending comma
            for time_current in line:
                f.write(str(time_current))
                if not end:
                    f.write(',')
                    end += 1
                else:
                    f.write('\n')
        print('Converted finished. Saved to '+output_file)

def get_i_t_energylab(source_file, loop_period, num_loop):
    """
    Seperate each loop from Energylab output files, each file contains all loops.
    Extract current and time from each loop and rewrite to a csv file.
    """
    with open(source_file, 'rt') as f:
        reader = csv.reader(f)
        data_list = list(reader)
        time_list = list(map(float, [row[0] for row in data_list][4:]))
        current_list = list(map(float, [row[2] for row in data_list][4:]))
    time0 = time_list[0]
    for loop in range(num_loop):
        time_write = []
        current_write = []
        for i, time in enumerate(time_list):
            if time >= (time0 + loop * loop_period) and time < (time0 + (loop + 1) * loop_period):
                time_write.append(time)
                current_write.append(current_list[i])
        with open(source_file[:-4]+'_'+str(loop + 1)+'.csv', 'w') as f:
            for i, time in enumerate(time_write):
                f.write(str(time))
                f.write(',')
                f.write(str(current_write[i]))
                f.write('\n')


if __name__ == '__main__':
    # rootdir = '/mnt/t/college/last/finaldesign/ENML/data/CA_ascii/20171108_p_curve'
    rootdir = 'T:/college/last/finaldesign/ENML/data/CA_ascii/20171108_p_curve'
    ivium = 1
    count = 0
    # for filename, subdir in walker(rootdir, re.compile('training(.*?)(\d+)$')):
    for filename, subdir in walker(rootdir):
        print('Opening '+filename)
        if ivium:
            get_i_t_ivium(subdir+'/'+filename, subdir+'/'+filename+'.csv')
        else:
            get_i_t_energylab(subdir+'/'+filename, loop_period=8, num_loop=30)
        count += 1
    print('#Files :%d'%count)