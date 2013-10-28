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
#install.packages("mvtnorm"), use for generating Bivariate Normal rv.
#install.packages("boot"), use for inverse logit function
#install.packages("MCMCpack"), MCMC package 

library(mvtnorm)
library(boot)
library(MCMCpack)

#get the data
data<-read.csv(file=paste("data/blr_data_",sim_num,".csv",sep=""))
beta<-read.csv(file=paste("data/blr_pars_",sim_num,".csv",sep=""))

#define the parameters used in the posterior
mu<-c(0,0)
cov<-diag(1,2)
x<-cbind(data$X1,data$X2)
y<-matrix(data$y)
mi<-matrix(data$n)
n=100
n.accept <- 0  # Track tha acceptance rate:
n.accept.burn<-0
niter<-10000
burnin<-1000
B.curr<-t(c(0,0))# Starting state:
v <- diag(1,2)#cov matrx for proposed distr.
B.draws <- matrix(NA,niter+burnin,2)# Store the samples
tune<-1.5


#log,target density function#
logfy<-function(B){dbinom(y,mi,inv.logit(x%*%B),log=TRUE)} # distribution of y given beta log scale
logfB<-function(B){dmvnorm(B,mu,cov,log=TRUE)} #the distribution of beta
log.target.density<-function(B){logfB(B)+sum(logfy(t(B)))} # the posterior distr.


for (i in 1:(niter+burnin)){
    B.prop <- t(mvrnorm(n=1,B.curr,v))# Propose a new state: 
    log.alpha <- log.target.density(B.prop)-log.target.density(B.curr)# Compute the log-acceptance probability:
    log.u <- log(runif(1))# Decide whether to accept or reject:
    if (log.u < log.alpha){B.curr <- B.prop} 
    if(i>burnin & log.u < log.alpha){n.accept<-n.accept+1} 
    B.draws[i,] <- B.curr# Store the current state
    if(i<(burnin+1) & log.u < log.alpha){n.accept.burn<-n.accept.burn+1}#count the burnin samples
    if(i==(.2)*burnin|i==(.5)*burnin|i==(.75)*burnin|i==(1.0)*burnin){
      if(n.accept.burn/i> .40){v<-exp(tune)*v}
      else{if(n.accept.burn/i<.20)
      {v<-v/exp(tune)}}
      print(cat(paste("Burnin Acceptance rate was  ",100*round(n.accept.burn/i,2),"%",sep="")))
      flush.console()}
    if(i>burnin & i%%1000==0){print(cat(paste("Acceptance rate was ",100*round(n.accept/(i-burnin),2),"%",sep="")))
     flush.console()
  }
}

#remove the burnin
B.draws<-B.draws[(burnin +1):(burnin+niter),]

#Get information about your run

cat(paste("Acceptance rate was ",100*round(n.accept/niter,2),"%\n",sep="")) 
acf(B.draws)
ess <- effectiveSize(B.draws)
cat("ESS:\n") ; print(ess)

B.draws<-mcmc(B.draws)

percents <- c(1:99)/100
B.percentile <- apply(B.draws,2,quantile,probs=percents)

#write.table(B.percentile,file = paste("results/blr_res_",sim_num,".csv",sep=""),sep=",",col.names=F, row.names=F)

write.table(B.percentile,file = paste("blr_res_",sim_num,".csv",sep=""),sep=",",col.names=F, row.names=F)

