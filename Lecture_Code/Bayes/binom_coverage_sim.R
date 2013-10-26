
# Binomial Interval Simulation

rm(list=ls())

# Number of datasets from which to compute the coverage:
n_datasets <- 500
alpha <- 0.05 # 95% intervals
# Jeffreys prior:
a <- 0.5
b <- 0.5
# Set n and p:
n <- 100
p <- 0.5

# Simulate n_datasets:
y_successes <- rbinom(n=n_datasets,size=n,prob=p)

# Computer lower value of posterior intervals:
pi_lo <- qbeta(alpha/2,a+y_successes,b+n-y_successes)

# Computer upper value of posterior intervals:
pi_hi <- qbeta(1-alpha/2,a+y_successes,b+n-y_successes)

# See if the true value is in the interval:
coverage_indicators <- (p >= pi_lo) & (p <= pi_hi)

# Compute the empirical coverage:
empirical_coverage <- mean(coverage_indicators)

cat(paste0("Empirical coverage of ",100*(1-alpha),"% Posterior Interval (with p=",p,"): ",empirical_coverage,"\n"))

# Functionize:
"binom.coverage" <- function(n,a,b,n_datasets,alpha,interval.type)
{
  y_successes <- rbinom(n=n_datasets,size=n,prob=p)
  if (interval.type=="conjugate.bayes"){
    pi_lo <- qbeta(alpha/2,a+y_successes,b+n-y_successes)
    pi_hi <- qbeta(1-alpha/2,a+y_successes,b+n-y_successes)
  } else {
    if (interval.type=="exact.freq"){
      pi_lo <- pi_hi <- rep(NA,n_datasets)
      for (i in 1:n_datasets){
        # binom.test is not vectorized :(
        tmp <- as.numeric(binom.test(x=y_successes,n=n,conf.level=1-alpha)$conf.int)
        pi_lo[i] <- tmp[1]
        pi_hi[i] <- tmp[2]
      } 
    } else {
      stop("Unknown interval type: must be either 'conjugate.bayes' or 'exact.freq'")
    }
  }
  coverage_indicators <- !((p < pi_lo) | (p > pi_hi))
  empirical_coverage <- mean(coverage_indicators)
  return(empirical_coverage)
}


binom.coverage(n_datasets=n_datasets,n=n,a=a,b=b,alpha=alpha,interval.type="conjugate.bayes")

n_datasets <- 100000
binom.coverage(n_datasets=n_datasets,n=n,a=a,b=b,alpha=alpha,interval.type="conjugate.bayes")

n_datasets <- 10000
alpha_vec <- seq(0,1,by=0.01)
coverage_vec <- rep(NA,length(alpha_vec))
for (i in 1:length(coverage_vec)){
  coverage_vec[i] <- binom.coverage(n_datasets=n_datasets,n=n,a=a,b=b,alpha=alpha_vec[i],interval.type="conjugate.bayes")
  cat("Finished coverage for alpha=",alpha_vec[i],"\n")
}

plot(x=1-alpha_vec,y=coverage_vec,type="p",xlab="Nominal Coverage",ylab="Actual Coverage",
     main="Nominal vs. Actual Coverage: Jeffreys Prior for Binomial")
abline(a=0,b=1,col='blue')
"full_coverage_sim" <- function(n_datasets,n,a,b,interval.type,main=NULL,do.plot=TRUE,verbose=TRUE)
{
  alpha_vec <- seq(0,1,by=0.01)
  coverage_vec <- rep(NA,length(alpha_vec))
  for (i in 1:length(coverage_vec)){
    coverage_vec[i] <- binom.coverage(n_datasets=n_datasets,n=n,a=a,b=b,alpha=alpha_vec[i],interval.type="conjugate.bayes")
    if (verbose){
      cat("Finished coverage for alpha=",alpha_vec[i],"\n")
    }
  }
  if (do.plot){
    if (is.null(main)){
      main <- "Nominal vs. Actual Coverage"
    }
    plot(x=1-alpha_vec,y=coverage_vec,type="p",xlab="Nominal Coverage",ylab="Actual Coverage",main=main)
    abline(a=0,b=1,col="red",lwd=1.5)
  }
  return(data.frame("nominal"=1-alpha_vec,"actual"=coverage_vec))
}

n <- 100
jeffreys <- full_coverage_sim(n_datasets=n_datasets,n=n,a=a,b=b,interval.type="conjugate.bayes",main=paste0("Jeffreys Prior (n=",n,")"))
exactfrq <- full_coverage_sim(n_datasets=n_datasets,n=n,a=a,b=b,interval.type="exact.freq",main=paste0("Exact Interval (n=",n,")"))


n <- 10000
jeffreys <- full_coverage_sim(n_datasets=n_datasets,n=n,a=a,b=b,interval.type="conjugate.bayes",main=paste0("Jeffreys Prior (n=",n,")"))
exactfrq <- full_coverage_sim(n_datasets=n_datasets,n=n,a=a,b=b,interval.type="exact.freq",main=paste0("Exact Interval (n=",n,")"))

lwd <- 1.5
plot(x=jeffreys$nominal,y=jeffreys$actual,type="l",xlab="Nominal",ylab="Actual",main=paste0("Coverage Plot: Binomial (n=",n,")"),col="red",lwd=lwd)
lines(x=exactfrq$nominal,y=exactfrq$actual,col="blue",lwd=lwd)
legend(legend=c("Jeffreys","Exact"),"topleft",col=c("red","blue"),lty=1)
abline(a=0,b=1,col="black",lwd=lwd)

