
# Demo python script to do random stuff...

# Run using e.g.,:
# python boot_camp_demo.py -o out_1.txt
#
# For batch jobs:
# sarray ./boot_camp_sarray.sh 
# squeue

import sys, getopt
import random
import urllib2

def main(argv):
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ofile="])
    except getopt.GetoptError:
        print 'boot_camp_demo.py -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'boot_cmap_demo.py -o <outputfile>'
            sys.exit()
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print 'Output file is "', outputfile
    return {'output': outputfile}


if __name__ == "__main__":
   specs = main(sys.argv[1:])

# Get GitHub README:
gh_readme = urllib2.urlopen("https://github.com/STA250/Stuff/blob/master/README.md")

# html should be a long string:
html = gh_readme.read()

# Find length:
html_len = len(html)

npars = 0
pars = {}
for i in range(3,html_len):
    if html[(i-3):i] == "<p>":
        print "Found <p> tag at i=" + str(i) + "...\n"
        for j in range(i,html_len):
            if html[j:(j+4)] == "</p>":
                print "Found </p> tag at j=" + str(j) + "...\n"
                pars[npars] = html[i:j]
                npars = npars+1
                break

# Randomly pick one of the paragraphs to write to file...
outfile = specs['output']
if len(outfile) == 0:
    outfile = "out.txt"

print "Writing output to: " + str(outfile)

if len(pars) > 0:
    # write to file...
    with open(outfile, 'a') as out_file:
        out_file.write(pars[random.randint(0,npars-1)])
else:
    # write stock message to file
    with open(outfile, 'a') as out_file:
        out_file.write("Couldn't find any pars in the supplied URL. Oh well. :/\n")

print "Signing off... :)"
sys.exit()



