
mini <- FALSE

# Set working directory:
setwd("/home/pdbaines/STA250/")
pkgpath <- "/home/pdbaines/R/x86_64-pc-linux-gnu-library/3.0/"

# Load packages:
library(BH,lib.loc=pkgpath)
library(bigmemory.sri,lib.loc=pkgpath)
library(bigmemory,lib.loc=pkgpath)
library(biganalytics,lib.loc=pkgpath)

# I/O specifications:
if (mini){
	rootfilename <- "blb_log_reg_mini"
} else {
	rootfilename <- "blb_log_reg_data"
}
datapath <- "/home/pdbaines/data"

# file specs:
infilename <- paste0(rootfilename,".txt")
backingfilename <- paste0(rootfilename,".bin")
descriptorfilename <- paste0(rootfilename,".desc")

# Set up I/O stuff:
infile <- paste(datapath,infilename,sep="/")
backingfile <- paste(datapath,backingfilename,sep="/")
descriptorfile <- paste(datapath,descriptorfilename,sep="/")

cat("Running read.big.matrix (i.e., creating file-backed matrix)...\n")

##
#
# Prepare filebacked big matrix file:
# 
# Note: 
# You only need to do this once, to create the .bin and .desc
# files. For the homework, these have already been 
# created for you, so you can just skip to the 
# attach.big.matrix portion below.
#
##

rbm.readtime <- system.time({
   goo <- read.big.matrix(infile, type="double", header=FALSE,
                          backingpath=datapath,
                          backingfile=backingfilename,
                          descriptorfile=descriptorfilename)
})
 
# Report read time:
cat("read.big.matrix() read time: ",round(rbm.readtime["elapsed"],4),"\n")
 
# Attach big.matrix :
if (verbose){
        cat("Attaching big.matrix...\n")
}
dat <- attach.big.matrix(dget(descriptorfile),backingpath=datapath)

# dataset size (6m):
n <- nrow(dat)

# number of covariates (last column is response):
d <- ncol(dat)-1

# First row:
dat[1,]

# Number of rows:
nrow(dat)

# First two rows, first ten columns:
dat[1:2,1:10]

# Random sample of 5 rows:
dat[sample(1:nrow(dat),5),]

# Try sampling more rows and seeing how long it takes
# (using e.g., system.time({...}))

