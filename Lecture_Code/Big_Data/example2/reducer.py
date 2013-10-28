#!/usr/bin/env python

from operator import itemgetter
import sys

# Hideously ugly reduce example

current_sum = 0.0
current_group = None
current_count = 0
verbose = False

# input comes from STDIN
for line in sys.stdin:
    # Remove trailing '\n'
    line = line.strip()
    # Extract (key,value)
    vs = line.split('\t')
    if (len(vs) != 2):
        print "vs not of length 2: " + str(vs)
    else:
        tmp_group = int(vs[0])
        if tmp_group == current_group:
            current_count = current_count + 1
            current_sum = current_sum + float(vs[1])
        else:
            if (current_count != 0):
                current_mean = current_sum / float(current_count)
            else:
                current_mean = float("inf")
            if not (current_group is None):
                if verbose:
                    print "group: " + str(current_group) + "; sum: " + str(current_sum) + "; count: " + str(current_count) + "; mean: " + str(current_mean)
                out_str = '{:03n}'.format(current_group) + "\t" + str(current_mean)
                print out_str
            current_group = int(tmp_group)
            current_sum = float(vs[1])
            current_count = 1

# Last one:
if verbose:
    print "group: " + str(current_group) + "; sum: " + str(current_sum) + "; count: " + str(current_count) + "; mean: " + str(current_mean)
current_mean = current_sum / float(current_count)
out_str = '{:03n}'.format(current_group) + "\t" + str(current_mean)
print out_str



