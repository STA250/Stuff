
# STA 250 :: Bayesian Module Code

######################################################
# Gibbs sampling:
######################################################

# Bivariate normal example:

plot_to_file <- FALSE

sigma.1 <- 2.0
sigma.2 <- 1.0
rho <- 0.8

Sigma <- matrix(c(sigma.1^2,rho*sigma.1*sigma.2,rho*sigma.1*sigma.2,sigma.2^2),2,2)

mu.1 <- 0.5
mu.2 <- 1.5

mu <- matrix(c(mu.1,mu.2))

library(mvtnorm)

nsamples <- 10000

# Independent samples:
ind.samples <- rmvnorm(n=nsamples,mean=mu,sigma=Sigma)

if (plot_to_file){
  pdf("samples.pdf")
}
plot(ind.samples)
if (plot_to_file){
  dev.off()
}

library(MASS)

if (plot_to_file){
  pdf("samples_kde.pdf")
}
z <- kde2d(ind.samples[,1],ind.samples[,2])
contour(z)
if (plot_to_file){
  dev.off()
}

# Gibbs sampling:

burnin <- 1000
total.samples <- nsamples +  burnin

gibbs.samples <- matrix(NA,nrow=total.samples,ncol=2)

# Need a starting value for x[2]:
x2.t <- 2.0

for (i in 1:total.samples){
  x1.t <- rnorm(n=1, mean=mu.1 + rho*sigma.1*(x2.t - mu.2)/sigma.2, sd=sqrt(1-rho^2)*sigma.1)
  x2.t <- rnorm(n=1, mean=mu.2 + rho*sigma.2*(x1.t - mu.1)/sigma.1, sd=sqrt(1-rho^2)*sigma.2)
  gibbs.samples[i,] <- c(x1.t,x2.t)
}

head(gibbs.samples)

# Discard the "burn-in" period:
gibbs.samples <- gibbs.samples[(burnin+1):(total.samples),]

pv <- c(0.025,0.10,0.25,0.50,0.75,0.90,0.975)
gibbs.q <- apply(gibbs.samples,2,quantile,probs=pv)
ind.q   <- apply(ind.samples,2,quantile,probs=pv)

cat("Independent samples:\n")
print(ind.q)

cat("Dependent samples:\n")
print(gibbs.q)

if (plot_to_file){
  pdf("gibbs_01.pdf")
}
par(mfrow=c(1,2))
plot(ind.samples,main="Independent Samples")
points(x=mu.1,y=mu.2,col="red", pch=16, cex=2.0)
plot(gibbs.samples,main="Dependent Samples (Gibbs Sampling)")
points(x=mu.1,y=mu.2,col="red", pch=16, cex=2.0)
if (plot_to_file){
  dev.off()
}

if (plot_to_file){
  pdf("gibbs_02.pdf")
}
plot(gibbs.samples,type="n",main="Dependent Samples (Gibbs Sampling)")
for (i in 1:10){
  lines(x=gibbs.samples[c(i,i+1),1],y=gibbs.samples[c(i,i+1),2],col="red", cex=2.0)
  text(x=gibbs.samples[i+1,1],y=gibbs.samples[i+1,2],labels=as.character(i),col="blue")
  Sys.sleep(2)
}

plot(gibbs.samples,type="n",main="Dependent Samples (Gibbs Sampling)")
for (i in 1:500){
  lines(x=gibbs.samples[c(i,i+1),1],y=gibbs.samples[c(i,i+1),2],col="red", cex=2.0)
  text(x=gibbs.samples[i+1,1],y=gibbs.samples[i+1,2],labels=as.character(i+1),col="blue")
}
if (plot_to_file){
  dev.off()
}

# Lag-1 autocorrelation should be roughly rho^2
# Correlation between X1 and X2 should be roughly rho

acf(gibbs.samples)

acf(gibbs.samples,plot=FALSE)[1:2,]

rho^2

# Check for independent samples:

acf(ind.samples)

# Effective sample sizes:

library(MCMCpack)

effectiveSize(gibbs.samples)
effectiveSize(ind.samples)

###############################

# Impact of starting values...

# Gibbs sampling:

burnin <- 0
nsamples <- 1000
total.samples <- nsamples +  burnin

gibbs.samples <- matrix(NA,nrow=total.samples,ncol=2)

# Need a starting value for x[2]:
x2.t <- 1201228412820340214.1231

for (i in 1:total.samples){
  x1.t <- rnorm(n=1, mean=mu.1 + rho*sigma.1*(x2.t - mu.2)/sigma.2, sd=sqrt(1-rho^2)*sigma.1)
  x2.t <- rnorm(n=1, mean=mu.2 + rho*sigma.2*(x1.t - mu.1)/sigma.1, sd=sqrt(1-rho^2)*sigma.2)
  gibbs.samples[i,] <- c(x1.t,x2.t)
}

head(gibbs.samples)

# Discard the "burn-in" period:
gibbs.samples <- gibbs.samples[(burnin+1):(total.samples),]

pv <- c(0.025,0.10,0.25,0.50,0.75,0.90,0.975)
gibbs.q <- apply(gibbs.samples,2,quantile,probs=pv)
ind.q   <- apply(ind.samples,2,quantile,probs=pv)

cat("Independent samples:\n")
print(ind.q)

cat("Dependent samples:\n")
print(gibbs.q)

plot(mcmc(gibbs.samples))

######################################################
# MCMC using Metropolis/Metropolis-Hastings
######################################################

## Metropolis-Hastings Algorithm Code:

do.metropolis <- FALSE
do.MH <- TRUE

x <- 
c(3.551481,  4.081882,  1.710738, 19.015891,  5.841083,  1.538956,  6.176577,
  2.219391,  9.524752,  2.519310,  7.982154,  3.838856,  9.852137,  3.491567,
  4.559637,  4.429825,  1.599168,  9.018974,  3.318598,  5.815163)

"log.target.density" <- function(sigma.sq,x){
  n <- length(x)
  if (sigma.sq < 0){
    return(-Inf)
  } else {
    return(-(1+(n/2))*log(sigma.sq) - sum(x^2)/(2*sigma.sq))
  }
}

if (do.metropolis){

cat("\n=============================================\n")
cat("Implementing Metropolis Algorithm...\n")
cat("=============================================\n\n")

# Starting state:
sigma.sq.curr <- 1.0

# Track tha acceptance rate:
n.accept <- 0

# Proposal distribution: sigma.sq.prop ~ N(sigma.sq.curr, v^2)
v <- 1.0

niter <- 10000

# Store the samples:
sigma.sq.draws <- rep(NA,niter)

# Metropolis-Algorithm:

for (i in 1:niter){
  # Propose a new state:
  sigma.sq.prop <- rnorm(n=1,mean=sigma.sq.curr,sd=v)
  # Compute the log-acceptance probability: log(alpha) = log( pi(theta^prop)/pi(theta^curr) )
  log.alpha <- log.target.density(sigma.sq=sigma.sq.prop,x=x) - log.target.density(sigma.sq=sigma.sq.curr,x=x)
  # Decide whether to accept or reject:
  log.u <- log(runif(1))
  if (log.u < log.alpha){
    # Accept...
    sigma.sq.curr <- sigma.sq.prop
    n.accept <- n.accept + 1
  } else {
    # Reject...
  }
  # Store the current state:
  sigma.sq.draws[i] <- sigma.sq.curr
}

# Report the acceptance rate:
cat(paste("Acceptance rate was ",100*round(n.accept/niter,2),"%\n",sep=""))

# Reformat as an MCMC object (makes prettier graphs)
library(MCMCpack)
sigma.sq.draws <- mcmc(sigma.sq.draws)

plot(sigma.sq.draws)

# Autocorrelation?
acf(sigma.sq.draws)

# Effective sample size?
ess <- effectiveSize(sigma.sq.draws)
cat("Effective sample size:\n") ; print(ess)

# Not good... try changing v...

"sigma.sq.metropolis" <- function(nsamples,x,v)
{
  # Starting state:
  sigma.sq.curr <- 1.0
  # Track tha acceptance rate:
  n.accept <- 0
  # Store the samples:
  sigma.sq.draws <- rep(NA,nsamples)
  for (i in 1:nsamples){
    # Propose a new state:
    sigma.sq.prop <- rnorm(n=1,mean=sigma.sq.curr,sd=v)
    # Compute the log-acceptance probability: log(alpha) = log( pi(theta^prop)/pi(theta^curr) )
    log.alpha <- log.target.density(sigma.sq=sigma.sq.prop,x=x) - log.target.density(sigma.sq=sigma.sq.curr,x=x)
    # Decide whether to accept or reject:
    log.u <- log(runif(1))
    if (log.u < log.alpha){
      # Accept...
      sigma.sq.curr <- sigma.sq.prop
      n.accept <- n.accept + 1
    } else {
      # Reject...
    }
    # Store the current state:
    sigma.sq.draws[i] <- sigma.sq.curr
  }
  # Report the acceptance rate:
  cat(paste("Acceptance rate was ",100*round(n.accept/nsamples,2),"%\n",sep=""))
  return(mcmc(sigma.sq.draws))
}

niter <- 10000

cat("\n############ v = 1.5 ###########\n\n")
sigma.sq.draws <- sigma.sq.metropolis(nsamples=niter,x=x,v=1.5)
cat("ESS = ",effectiveSize(sigma.sq.draws),"\n") # about 25, acc ~ 95%

cat("\n############ v = 2.0 ###########\n\n")
sigma.sq.draws <- sigma.sq.metropolis(nsamples=niter,x=x,v=2.0)
cat("ESS = ",effectiveSize(sigma.sq.draws),"\n") # about 25, acc ~ 95%

cat("\n############ v = 10.0 ###########\n\n")
sigma.sq.draws <- sigma.sq.metropolis(nsamples=niter,x=x,v=10.0)
cat("ESS = ",effectiveSize(sigma.sq.draws),"\n") # about 520, acc ~ 75%

cat("\n############ v = 25.0 ###########\n\n")
sigma.sq.draws <- sigma.sq.metropolis(nsamples=niter,x=x,v=25.0)
cat("ESS = ",effectiveSize(sigma.sq.draws),"\n") # about 1250, acc ~ 55%

cat("\n############ v = 36.0 ###########\n\n")
sigma.sq.draws <- sigma.sq.metropolis(nsamples=niter,x=x,v=36.0)
cat("ESS = ",effectiveSize(sigma.sq.draws),"\n") # about 1250, acc ~ 45%

cat("\n############ v = 50.0 ###########\n\n")
sigma.sq.draws <- sigma.sq.metropolis(nsamples=niter,x=x,v=50.0)
cat("ESS = ",effectiveSize(sigma.sq.draws),"\n") # about 1640, acc ~ 35%

cat("\n############ v = 100.0 ###########\n\n")
sigma.sq.draws <- sigma.sq.metropolis(nsamples=niter,x=x,v=100.0)
cat("ESS = ",effectiveSize(sigma.sq.draws),"\n") # about 1050, acc ~ 20%

cat("\n############ v = 500.0 ###########\n\n")
sigma.sq.draws <- sigma.sq.metropolis(nsamples=niter,x=x,v=500.0)
cat("ESS = ",effectiveSize(sigma.sq.draws),"\n") # about 300, acc ~ 4%

# Usually works best when acceptance rate is between 30-60%

# Lets use the best one:
cat("\n############ v = 50.0 ###########\n\n")
sigma.sq.draws <- sigma.sq.metropolis(nsamples=niter,x=x,v=50.0)

plot(sigma.sq.draws)
print(summary(sigma.sq.draws))

# Compare to what the answer should be more directly:
library(geoR)
print(quantile(rinvchisq(n=1000000,df=length(x),scale=mean(x^2)),prob=c(0.025,0.25,0.50,0.75,0.975)))

# Not bad... probably want a larger number of iterations though!

} 

###############################################################################

if (do.MH){

cat("\n=============================================\n")
cat("Implementing Metropolis-Hastings Algorithm...\n")
cat("=============================================\n\n")

# Lets do Metropolis-Hastings now (i.e., non-symmetric proposal distribution):

"sigma.sq.mh" <- function(nsamples,x,v,dbg=FALSE)
{
  # Starting state:
  sigma.sq.curr <- 1.0
  # Track tha acceptance rate:
  n.accept <- 0
  # Store the samples:
  sigma.sq.draws <- rep(NA,nsamples)

  for (i in 1:nsamples){

    # Propose a new state:
    sigma.sq.prop <- rgamma(n=1,shape=v*sigma.sq.curr,rate=v) # Mean = sigma.sq.curr, non-symmetric

    # Compute the log-acceptance probability:
    # log(alpha) = log(pi(theta^prop)*g(theta^curr|theta^prop)/(pi(theta^curr)*g(theta^prop|theta^curr)) )
    #            = log(pi(theta^prop)) + log(g(theta^curr|theta^prop)
    #               - log(pi(theta^curr)) + log(g(theta^prop|theta^curr)

    p.lprop <- log.target.density(sigma.sq=sigma.sq.prop,x=x)
    p.lcurr <- log.target.density(sigma.sq=sigma.sq.curr,x=x)
    g.lpc <- dgamma(sigma.sq.curr,shape=v*sigma.sq.prop,rate=v,log=TRUE)
    g.lcp <- dgamma(sigma.sq.prop,shape=v*sigma.sq.curr,rate=v,log=TRUE)

    log.alpha <- log.target.density(sigma.sq=sigma.sq.prop,x=x) - 
                 log.target.density(sigma.sq=sigma.sq.curr,x=x) +
                 dgamma(sigma.sq.curr,shape=v*sigma.sq.prop,rate=v,log=TRUE) -  # Note: dgamma(...,log=TRUE)
                 dgamma(sigma.sq.prop,shape=v*sigma.sq.curr,rate=v,log=TRUE)    #  **NOT** log(dgamma(...)) !!!

    # Little extra check to catch any possible NA's:
    if(is.na(log.alpha)){
      log.alpha <- -Inf
    }

    if (dbg){
      # Error in computing log(alpha), debug:
      cat(paste("sigma.sq.curr = ",sigma.sq.curr,"\n",sep=""))
      cat(paste("sigma.sq.prop = ",sigma.sq.prop,"\n",sep=""))
      cat(paste("log(pi(sigma.sq.prop)) = ",p.lprop,"\n",sep=""))
      cat(paste("log(pi(sigma.sq.curr)) = ",p.lcurr,"\n",sep=""))
      cat(paste("log(g(sigma.sq.curr|sigma.sq.prop)) = ",g.lpc,"\n",sep=""))
      cat(paste("log(g(sigma.sq.prop|sigma.sq.curr)) = ",g.lcp,"\n",sep=""))
      cat(paste("log(alpha) = ",log.alpha,"\n",sep=""))
    }

    # Decide whether to accept or reject:
    log.u <- log(runif(1))
    if (log.u < log.alpha){
      # Accept...
      sigma.sq.curr <- sigma.sq.prop
      n.accept <- n.accept + 1
    } else {
      # Reject...
    }
    # Store the current state:
    sigma.sq.draws[i] <- sigma.sq.curr
  }
  # Report the acceptance rate:
  cat(paste("Acceptance rate was ",100*round(n.accept/nsamples,2),"%\n",sep=""))
  return(mcmc(sigma.sq.draws))
}

niter <- 10000

cat("\n############ v = 1.5 ###########\n\n")
sigma.sq.draws <- sigma.sq.mh(nsamples=niter,x=x,v=1.0)
cat("ESS = ",effectiveSize(sigma.sq.draws),"\n") # about 25, acc ~ 95%

plot(sigma.sq.draws)

v.candidates <- c(0.001,0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.1,0.2,0.25)
ess.values <- rep(NA,length(v.candidates))

for (i in 1:length(v.candidates)){
  cat(paste("\n############ v = ",v.candidates[i]," ###########\n\n",sep=""))
  sigma.sq.draws <- sigma.sq.mh(nsamples=niter,x=x,v=v.candidates[i])
  tmp.ess <- effectiveSize(sigma.sq.draws)
  cat("ESS = ",tmp.ess,"\n")
  ess.values[i] <- tmp.ess
}

# Plot ESS as a function of sample size:

plot(x=v.candidates,y=ess.values,xlab="v",ylab="ESS",lwd=1.6,
     type="b",main="Effective Sample Size vs. Tuning Parameter (v) for MH")

sigma.sq.draws <- sigma.sq.mh(nsamples=250000,x=x,v=0.02) # Best v

# Should really throw away burnin too... :)

#plot(sigma.sq.draws)
print(summary(sigma.sq.draws))

# Compare to what the answer should be more directly:
library(geoR)
print(quantile(rinvchisq(n=1000000,df=length(x),scale=mean(x^2)),prob=c(0.025,0.25,0.50,0.75,0.975)))


}

"rmixnorm" <- function(n,p,mu,sd)
{
  x <- rep(NA,n)
  if (abs(sum(p)-1)>.Machine$double.eps){
    stop("'p' does not sum to one")
  }
  k <- length(p) # number of mixtures
  I <- sample(x=1:k,size=n,prob=p,replace=TRUE) # mixture indices
  x <- rnorm(n=n,mean=mu[I],sd=sd[I]) # sample components
  return(x)
}

"mh" <- function(nsamples,data,theta_start,burnin,
                 log_target,prop_func,log_prop_density,
                 tuning_pars,dbg=FALSE)
{
  # Track tha acceptance rate:
  n.accept <- 0
  # Store the samples:
  p <- length(unlist(theta_start))
  draws <- matrix(NA,nrow=nsamples,ncol=p)
  colnames(draws) <- names(unlist(theta_start))
  draws <- mcmc(draws)
  theta_curr <- theta_start
  
  for (i in 1:(burnin+nsamples)){
    
    # Propose a new state:
    theta_prop <- prop_func(theta_curr,tuning_pars)
    
    # Compute transition probabilities:
    log_prop_to_curr <- log_prop_density(to=theta_curr,from=theta_prop,tuning_pars=tuning_pars)
    log_curr_to_prop <- log_prop_density(to=theta_prop,from=theta_curr,tuning_pars=tuning_pars)
    
    # Compute the value of target density:
    log_prop <- log_target(theta_prop,data)
    log_curr <- log_target(theta_curr,data)
    
    # Compute the log-acceptance probability:
    # log(alpha) = log(pi(theta^prop)*g(theta^curr|theta^prop)/(pi(theta^curr)*g(theta^prop|theta^curr)) )
    #            = log(pi(theta^prop)) + log(g(theta^curr|theta^prop)
    #               - log(pi(theta^curr)) + log(g(theta^prop|theta^curr)
    log_alpha <- log_prop - log_curr + log_prop_to_curr - log_curr_to_prop
    
    # Little extra check to catch any possible NA's:
    if (is.na(log_alpha)){
      log_alpha <- -Inf
    }
    
    if (dbg){
      # Error in computing log(alpha), debug:
      cat("theta_curr:\n") ; print(theta_curr)
      cat("theta_prop:\n") ; print(theta_prop)
      cat(paste("log(pi(theta_prop)) = ",log_prop,"\n",sep=""))
      cat(paste("log(pi(theta_curr)) = ",log_curr,"\n",sep=""))
      cat(paste("log(p(prop --> curr)) = ",log_prop_to_curr,"\n",sep=""))
      cat(paste("log(g(curr --> prop)) = ",log_curr_to_prop,"\n",sep=""))
      cat(paste("log(alpha) = ",log_alpha,"\n",sep=""))
    }
    
    # Decide whether to accept or reject:
    log_u <- log(runif(1))
    if (log_u < log_alpha){
      # Accept...
      theta_curr <- theta_prop
      n.accept <- n.accept + 1
    } else {
      # Reject...
    }
    if (i > burnin){
      # Store the current state:
      draws[i-burnin,] <- unlist(theta_curr)
    }
  }
  # Report the acceptance rate:
  cat(paste("Acceptance rate was ",100*round(n.accept/nsamples,2),"%\n",sep=""))
  return(draws)
}


# Generate data:
n <- 100
theta <- log(1)
y <- rnorm(n=n,mean=exp(theta),sd=1)

"example_log_post" <- function(theta,y)
{
    return(-0.5*((theta^2) + sum((y-exp(theta))^2)))
}

"example_prop" <- function(theta,tuning_pars)
{
    return(rnorm(n=1,mean=theta,sd=sqrt(tuning_pars)))
}

"example_log_trans" <- function(from,to,tuning_pars)
{
    return(0.0)
}

nsamples <- 10000
burnin <- 5000
theta_0 <- 0.0
example_tuning_pars <- 1.0 # v^{2}
example_dbg <- FALSE

draws <- mh(nsamples=nsamples,
            data=y,
            theta_start=theta_0,
            burnin=burnin,
            log_target=example_log_post,
            prop_func=example_prop,
            log_prop_density=example_log_trans,
            tuning_pars=example_tuning_pars,
            dbg=example_dbg)

summary(draws)
acf(draws,plot=FALSE)
effectiveSize(draws)
