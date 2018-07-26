# Extract required lines from given ascii files
# and reclassify them by potential

import os
from dir_walker import walker

# rootdir = 'G:/finaldesign/ENML/data/20180509'
rootdir = "C:/code/ENML/data/20180523"
save_dir = 'C:/code/ENML/data/20180523_extracted'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# extract
for filename, subdir in walker(rootdir):
    print("Opening "+filename)
    a = []
    with open(subdir+'/'+filename, 'rb') as f:
        for line in f:
            try:
                a.append(line.decode())
            except UnicodeDecodeError:
                a.append('\r\n')
                pass
        # if not a[75].split()[0] == '8000':
        #     # judge whether the file is correct
        #     print("File "+filename+' is not correct')
        #     continue
        extracted_lines = a[76:-2]

    # save
    with open(save_dir+'/'+filename, 'w', newline='\n') as f:
        f.write('time/s  I/A\r\n')
        for line in extracted_lines:
            f.write(line)
