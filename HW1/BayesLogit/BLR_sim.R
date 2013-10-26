
##
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##

library(mvtnorm) # For rmvnorm

"sim_logreg" <- function(m,p,beta.0,Sigma.0.inv)
{
  n <- rpois(n=m,lambda=100)
  X <- matrix(1,ncol=1,nrow=m)
  if (p>=2){
    for (i in 2:p){
      X <- cbind(X,rnorm(m))      
    }
  }
  beta <- matrix(rmvnorm(n=1,mean=beta.0,sigma=solve(Sigma.0.inv)),ncol=1)
  eta <- X%*%beta
  y <- rbinom(n=m,size=n,prob=exp(eta)/(1+exp(eta)))
  return(list("y"=y,"X"=X,"n"=n,"beta"=beta))
}

########################################################################################
########################################################################################
## Handle batch job arguments:

# 1-indexed version is used now.
args <- commandArgs(TRUE)

cat(paste0("Command-line arguments:\n"))
print(args)

####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start <- 1000
length.datasets <- 200
#######################

if (length(args)==0){
  sinkit <- FALSE
  sim_num <- sim_start + 1
  set.seed(1330931)
} else {
  # Sink output to file?
  sinkit <- TRUE
  # Decide on the job number, usually start at 1000:
  sim_num <- sim_start + as.numeric(args[1])
  # Set a different random seed for every job number!!!
  set.seed(762*sim_num + 1330931)
}

# Simulation datasets numbered 1001-1200

########################################################################################
########################################################################################


#################################################
m <- 100
p <- 2
beta.0 <- matrix(c(0,0))
Sigma.0.inv <- diag(rep(1.0,p))
#################################################

outdir <- "data/"
outfile_data <- paste0(outdir,"blr_data_",sim_num,".csv")
outfile_pars <- paste0(outdir,"blr_pars_",sim_num,".csv")

dat <- sim_logreg(m=m,p=2,beta.0=beta.0,Sigma.0.inv=Sigma.0.inv)
y <- dat$y
n <- dat$n
X <- dat$X
colnames(X) <- paste0("X",1:p)
beta <- dat$beta
names(beta) <- paste0("beta",1:p)

write.table(data.frame("y"=y,"n"=n,X),file=outfile_data,sep=",",quote=FALSE,row.names=FALSE,col.names=TRUE)
write.table(data.frame(beta),file=outfile_pars,sep=",",quote=FALSE,row.names=FALSE,col.names=TRUE)




