import os
import re
import csv
from dir_walker import walker

def get_save_path(save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = save_dir+'/'+filename
    return path

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

def get_i_v_t_energylab(dir, filename, loop_period, num_loop):
    """
    Seperate each loop from Energylab output files, each file contains all loops.
    Extract current, potential and time from each loop and rewrite to a csv file.
    Notice that loop_period is counted by s.
    """
    source_file = dir+'/'+filename
    with open(source_file, 'rt') as f:
        reader = csv.reader(f)
        data_list = list(reader)
        time_list = list(map(float, [row[0] for row in data_list[4:]]))
        current_list = list(map(float, [row[2] for row in data_list[4:]]))
        voltage_list = list(map(float, [row[1] for row in data_list[4:]]))
    time0 = time_list[0]
    coordinate = list(range(loop_period * 1000)) # ms
    for loop in range(num_loop):
        time_write = []
        current_write = []
        voltage_write = []
        for i, time in enumerate(time_list):
            if time >= (time0 + loop * loop_period) and time < (time0 + (loop + 1) * loop_period):
                time_write.append(time)
                current_write.append(current_list[i])
                voltage_write.append(voltage_list[i])
        with open(get_save_path(dir+'/i_t_csv', filename[:-4]+'_'+str(loop + 1)+'.csv'), 'w') as f:
            f.write('time/s,I/A')
            f.write('\n')
            for i, current in enumerate(current_write):
                f.write(str(coordinate[i]))
                f.write(',')
                f.write(str(current))
                f.write('\n')
        with open(get_save_path(dir+'/e_t_csv', filename[:-4]+'_'+str(loop + 1)+'.csv'), 'w') as f:
            f.write('time/s,voltage/V')
            f.write('\n')
            for i, voltage in enumerate(voltage_write):
                f.write(str(coordinate[i]))
                f.write(',')
                f.write(str(voltage))
                f.write('\n')


if __name__ == '__main__':
    # rootdir = '/mnt/t/college/last/finaldesign/ENML/data/CA_ascii/20171108_p_curve'
    # rootdir = 'T:/college/last/finaldesign/ENML/data/CA_ascii/20171228/energylab'
    rootdir = 'C:/code/ENML/data/20180407_r/raw'
    ivium = 1
    count = 0
    # for filename, subdir in walker(rootdir, re.compile('training(.*?)(\d+)$')):

    # Notice: Because of the re: first time this works, but the second time, 
    #         generated files included, and it fails. Personally it's not a big deal, 
    #         so there's no plan to fix this yet.
    for filename, subdir in walker(rootdir):
        print('Opening '+filename)
        if ivium:
            get_i_t_ivium(subdir+'/'+filename, subdir+'/'+filename+'.csv')
        else:
            get_i_v_t_energylab(subdir, filename, loop_period=10, num_loop=30)
        count += 1

    print('#Files :%d'%count)