import csv
import sys
import numpy as np

## 
#    filename: name of csv file
#     indices: array of row numbers to select
#          nr: number of rows to sample i.e., len(indices)
#          nc: number of cols in file
# print_every: number of lines per status update
#     verbose: True/False, level of verbosity
##

def read_some_lines_csv(filename,indices,nr,nc,n,print_every=1000,verbose=False):
	# Storage:
	subset = np.empty(nr*nc)
	subset.shape = (nr,nc)
	# Read file and extract selected rows:
	row_num = 0
	sampled_so_far = 0
	# utility stuff:
	next_ix = indices[0]
	# append value to avoid oob:
	indices = np.append(indices,n+1)
	with open(filename, 'rb') as csvfile:
		datareader = csv.reader(csvfile, delimiter=',')
		for current_row in datareader:
			# Check if needed:
			if row_num == next_ix:
				# Yup:
				if verbose:
					print "Sampled row " + str(row_num) + ":"
					print current_row
				# Store values:
				subset[sampled_so_far,:] = current_row[:]
				# Increment indices:
				sampled_so_far += 1
				next_ix = indices[sampled_so_far]
			# Increment row counter:
			row_num += 1
			# Print progress?
			if verbose and ((row_num % print_every) == 0):
				print "Finished row " + str(row_num) + " (sampled " + str(sampled_so_far) + "/" + str(b) + " rows so far)"
	return subset



