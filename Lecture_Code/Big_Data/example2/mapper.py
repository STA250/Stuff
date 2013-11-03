#!/usr/bin/env python

import sys

# Actually does nothing, just ...

# input comes from STDIN (standard input)
for line in sys.stdin:
    # split the line into (group, value)
    line = line.strip() # Need to remove trailing '\n'
    entries = line.split('\t')
    out_str = str(entries[0]) + str('\t') + str(entries[1])
    print out_str

