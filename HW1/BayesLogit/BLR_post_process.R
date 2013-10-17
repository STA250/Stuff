##
## Post-process a large-scale validation run:
##

# Start from clean workspace:
rm(list=ls())

# Load in validation functions:
source("validation_funcs.R")

## Handle batch job arguments:

# 1-indexed version is used now.
args <- commandArgs(TRUE)
#args <- 4

cat(paste0("Command-line arguments:\n"))
print(args)

# Read in all of the quantiles:
total_num_sims     <- 200
num_sims_completed <- 200
sim_start <- 1000

##################################
coverplot_name <- "coverage_line_plot.pdf"
resdir <- "results/"
pardir <- "data/"
verbose <- TRUE
###################################

## Read in the posterior quantiles and true parameter values:
qlist      <- list(NULL)
truth      <- list(NULL)
num_so_far <- 0

for (iter in 1:(num_sims_completed))
{
  num_so_far <- num_so_far+1
  sim_num <- sim_start + iter
  outfile_res <- paste0(resdir,"blr_res_",sim_num,".csv")
  outfile_par <- paste0(pardir,"blr_pars_",sim_num,".csv")
  qs <- try(read.csv(outfile_res,header=FALSE),silent=TRUE)
  ps <- try(read.csv(outfile_par,header=TRUE),silent=TRUE)

  if (class(qs)!="try-error")
  {
    if (any(dim(qs)!=c(99,2))){
      warning(paste0("File: ",outfile_res," does not have 99 rows and 2 columns!"))
      qlist[[num_so_far]] <- NULL
    } else {
      # Extract results:
      qlist[[num_so_far]] <- t(as.matrix(qs))
      rownames(qlist[[num_so_far]]) <- c("beta_0","beta_1")
    }  
  } else {
    qlist[[num_so_far]] <- NULL
  }

  if (class(ps)!="try-error")
  {
    if (any(dim(ps)!=c(2,1))){
      warning(paste0("File: ",outfile_par," does not have 2 rows and 1 column!"))
      truth[[num_so_far]] <- NULL
    } else {
      # Extract results:
      truth[[num_so_far]] <- as.numeric(ps[,1])
    }  
  } else {
    truth[[num_so_far]] <- NULL
  }
  
  if ( (iter%%100)==0 && verbose) {
    cat(paste0("Dataset ", iter, " has been read...\n"))
  }

  
} # END for loop over iter: num_sims_completed


cat("Computing Validation statistics of MCMC draws...\n")   

## Compute coverage proportions:

# Find first successful non-NULL qlist:
for (i in 1:(num_sims_completed)){
  if (!is.null(qlist[[i]])){
    all.par.names <- rownames(qlist[[i]])
    break
  }
}

coverage_out      <- cover_func(X=qlist,truth=truth,type='one-sided',return.type='all')
cover_indicators  <- coverage_out$indicators
prop_cover        <- coverage_out$proportions
successful_dataset_indices <- coverage_out$successful_dataset_indices
num_sims_successfully_completed <- length(successful_dataset_indices)

####
## logical(0) for datasets not analyzed
####

cat(paste0("\nThere were ",num_sims_successfully_completed," datasets that were successfully analyzed...\n"))

## Create nice looking table for results:
p_table <- prop_cover
cat("\nCoverage table:\n\n")
print(round(p_table,3))

## Print latex output in convenient form:
library(xtable)

p_keep <- c(1,5,10,25,50,75,90,95,99)
cov_summary <- round(t(p_table)[p_keep,],4)
rownames(cov_summary) <- sprintf("p_%02d",p_keep)

sink("coverage_summaries.tex")
print(xtable(cov_summary))
sink()

sink("coverage_summaries.txt")
print(cov_summary)
sink()

## Make some plots of the results:
actual_coverage  <- prop_cover
desired_coverage <- seq(0.01,0.99,by=0.01)

cat("Making global coverage plot...\n") 
pdf(coverplot_name)
global_coverage_plot(desired_coverage=desired_coverage,actual_coverage=t(actual_coverage),bands=TRUE,num_sims=num_sims_successfully_completed)
dev.off()

