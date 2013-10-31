##Homework 1, Problem 3

########################################################################################
########################################################################################
#install.packages("mvtnorm"), use for generating Bivariate Normal rv.
#install.packages("boot"), use for inverse logit function
#install.packages("emdbook"), use for multivariate normal density
#install.packages("MCMCpack"), MCMC packag
library(mvtnorm)
library(boot)
library(emdbook)
library(MCMCpack)

#get the data, change the categorical variables (M=1. B=0 in 11th column)
data<-read.table("breast_cancer.txt")
data<-data[-1,]


y<-matrix(rep(0,569))
for (i in 1:569){
  if (data$V11[i]=="M"){y[i,]<-1}
  else
  {y[i,]<-0}
}
data<-data[,-11]
data<-cbind(data,y)

#define the parameters used in the posterior
mu<-c(rep(0,11))
cov<-diag(1000,11)
x<-data[,-11]
one<-rep(1,569)
x<-cbind(one,x)
for (i in 1:11) {data[,i]<-scale(as.numeric(data[,i]))}#standardize x's
x<-cbind(data[,c(1:11)])
for (i in 1:11){x[,i]<-as.numeric(as.character(x[,i]))}
x<-as.matrix(x)



n.accept <- 0  # Track tha acceptance rate:
n.accept.burn<-0
niter<-3000
burnin<-1000
B.curr<-matrix(rep(0,11),11,1)# Starting state:

#cov matrx for proposed distr.
v<-matrix(0,11,11)
for (i in 1:11)
{v[i,i]<-var(x[,i])}


#create matrix for beta samples
B.draws <- matrix(NA,niter+burnin,11)# Store the samples
tune<-exp(1.01)



#log,target density function#
logfy<-function(B){dbinom(y,1,inv.logit(x%*%B),log=TRUE)} # distribution of y given beta log scale
logfB<-function(B){dmvnorm(t(B),mu,cov,log=TRUE)} #the distribution of beta
log.target.density<-function(B){logfB(B)+sum(logfy(B))} # the posterior distr.

##Metropolis Algorithm
for (i in 1:(niter+burnin)){
  B.prop <- mvrnorm(n=1,B.curr,v)# Propose a new state: 
  log.alpha <- log.target.density(B.prop)-log.target.density(B.curr)# Compute the log-acceptance probability:
  log.u <- log(runif(1))# Decide whether to accept or reject:
  if (log.u < log.alpha){B.curr <- B.prop} 
  if(i>burnin & log.u < log.alpha){n.accept<-n.accept+1} 
  B.draws[i,] <- B.curr# Store the current state
  if(i<(burnin+1) & log.u < log.alpha){n.accept.burn<-n.accept.burn+1}#count the burnin samples
  if(i==(.25)*burnin|i==(.5)*burnin|i==(.75)*burnin|i==(1.0)*burnin){
    if(n.accept.burn/i> .40){v<-v*(tune)}
    else{if(n.accept.burn/i<.20)
    {v<-v/(tune)}}
    print(cat(paste("Burnin Acceptance rate was  ",100*round(n.accept.burn/i,2),"%",sep="")))
    flush.console()}
  if(i>burnin & i%%1000==0){print(cat(paste("Acceptance rate was ",100*round(n.accept/(i-burnin),2),"%",sep="")))
                            flush.console()
  }
}

#remove the burnin
B.draws<-B.draws[(burnin +1):(burnin+niter),]
B.draws_check<-B.draws
#Get information about your run

cat(paste("Acceptance rate was ",100*round(n.accept/niter,2),"%\n",sep="")) 
acf(B.draws)
ess <- effectiveSize(B.draws)
cat("ESS:\n") ; print(ess)

B.draws<-mcmc(B.draws)

##get the quantiles, to figure out related covariates
percents <- c(1,5,25,50,75,95,99,100)/100
B.percentile <- apply(B.draws,2,quantile,probs=percents)

##posterior predictive checking of means.

y_check<-matrix(0,569,niter)
for(i in 1:niter){
  b<-as.matrix(B.draws_check[i,])
  y.curr<-as.matrix(rbinom(569,1,inv.logit(x%*%b)))
  y_check[,i]<-y.curr
}

png("means.png")
sample_means<-apply(y_check,2,mean)
true_mean<-mean(y)
hist(sample_means, main="Histogram of Sample Means")
abline(v=true_mean,col="red")
dev.off()

##get lag-1 autocorr

lag1 = sapply(1:11, function (i) acf(B.draws[,i],plot=F)$acf[2])

autotable = t(as.table(lag1))
colnamest<-c("\beta1","\beta2","\beta3","\beta4","\beta5","\beta6","\beta7","\beta8","\beta9","\beta10","\beta11")
colnames(autotable)<-colnamest











