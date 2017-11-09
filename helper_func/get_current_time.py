import os
import re
from dir_walker import walker

def get_i_t(source_file, output_file):
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
			for time_current in line:
				f.write(str(time_current))
				f.write(',')
			f.write('\n')
		print('Converted finished. Saved to '+output_file)

if __name__ == '__main__':
	rootdir = '/mnt/t/college/last/finaldesign/ENML/data/CA_ascii/20171108'
	count = 0
	for filename, subdir in walker(rootdir, re.compile('training(.*?)(\d+)$')):
		print('Opening '+filename)
		get_i_t(rootdir+'/'+filename, rootdir+'/'+filename+'.csv')
		count += 1
	print('#Files :%d'%count)