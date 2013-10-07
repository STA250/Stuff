
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

###############################################################################
# Bootstrap Code

n <- 250
mu <- c(-1,-0.5,0.9)
sigma <- c(0.1,0.5,0.2)
p <- c(0.2,0.4,0.4)
x <- rmixnorm(n=n,mu=mu,p=p,sd=sigma) 

library(MASS)
truehist(x,nbins=40)

# Estimate:
xbar <- mean(x)

"bootstrap" <- function(data,f,B=200)
{
  # assumes scalar estimates
  # and that data can be sample()'d from
  est_vec <- rep(NA,B)
  for (b in 1:B){
    # Resample dataset:
    x_star <- sample(data,replace=TRUE)
    # Compute estimate:
    est_vec[b] <- f(x_star)
  }
  # Return bootstrap distribution:
  return(est_vec)
}

# Perform bootstrap:
breps <- bootstrap(data=x,f=mean,B=500)
# Plot:
truehist(breps)
# Estimate SE:
sd(breps)

# Increase B:
breps <- bootstrap(data=x,f=mean,B=10000)
truehist(breps)
sd(breps)

###############################################################################
# Markov Chain Code
# Simple transition kernel: 
# X_{t+1} | x_{t} ~ N(\rho*x_{t},\sigma^{2})

rho <- 0.5
ss <- 1.0
n <- 100000
x0 <- 0.0 

x <- rep(NA,n)
x[1] <- x0
sigma <- sqrt(ss)

for (i in 2:n){
  x[i] <- rnorm(n=1,mean=rho*x[i-1],sd=sigma)
}

library(coda)
library(MASS)

truehist(x)
densplot(mcmc(x))

var(x) ; sd(x)




