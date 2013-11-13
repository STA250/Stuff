"blr.log.posterior" <- function(beta,n,y,X,beta.0,Sigma.0.inv)
{
  # Normal prior:
  lnp <- -0.5*crossprod(beta-beta.0,Sigma.0.inv)%*%(beta-beta.0)
  # Binomial terms:
  exp.eta <- exp(X%*%beta)
  lbt <- sum(dbinom(x=y,size=n,prob=exp.eta/(1+exp.eta),log=TRUE))
  # Total:
  ret <- lnp + lbt
  return(ret)
}

"tune" <- function(v,acc,iter,verbose=FALSE){
  # acc and v can both be vectors:
  # acc -- number of acceptances since last tune
  # iter -- number of iterations since last tune
  # v -- current vector of proposal sd's or var's
  if (verbose){
    cat("v (before tuning):\n") ; print(v)
    cat("acc (before tuning):\n") ; print(acc)
    cat("iter:\n") ; print(iter) 
  }
  for (i in 1:length(v)){
    if (acc[i]/iter > 0.50){ v[i] <- v[i]*2 } 
    if (acc[i]/iter > 0.70){ v[i] <- v[i]*4 } 
    if (acc[i]/iter > 0.80){ v[i] <- v[i]*6 } 
    if (acc[i]/iter > 0.95){ v[i] <- v[i]*10 } 
    if (acc[i]/iter < 0.30){ v[i] <- v[i]/2 } 
    if (acc[i]/iter < 0.20){ v[i] <- v[i]/4 } 
    if (acc[i]/iter < 0.05){ v[i] <- v[i]/10 } 
  }
  if (verbose){
    cat("v (after tuning):\n") ; print(v)
  }
  return(v)
}

"bayes.logreg" <- function(n,y,X,
                           beta.0,Sigma.0.inv, # prior mean and variance
                           algorithm, # "MH" or "MHwG"
                           niter=10000,burnin=1000,
                           print.every=1000,retune=100,verbose=TRUE)
{
  p <- ncol(X)
  # Starting value for beta
  #rlm <- glm(cbind(y,n-y)~0+X,family=binomial(link="logit"))
  rlm <- lm((y/n)~0+X)
  beta.t <- matrix(coef(rlm))
  if (algorithm=="MH"){
    # Tuning variance:
    k <- 10
    V <- vcov(rlm)
    acc <- 0
    acc.since.last.tune <- 0
  } else if (algorithm=="MHwG"){
    k <- rep(1,p)
    V <- sqrt(diag(vcov(rlm))) # SE's
    acc <- rep(0,p)
    acc.since.last.tune <- rep(0,p)
  } else {
    stop("Invalid 'algorithm': must be either 'MH' or 'MHwG'")
  }
  # Draws:
  beta.samples <- mcmc(matrix(NA,nrow=niter,ncol=p))
  varnames(beta.samples) <- colnames(X)
  iter.since.last.tune <- 0
  for (iter in 1:(niter+burnin)){
    if (iter%%retune == 0 && iter<=burnin){
      k <- tune(k,acc.since.last.tune,iter.since.last.tune,verbose)
      # Reset the tuning counters:
      acc.since.last.tune[] <- 0
      iter.since.last.tune <- 0
    }
    if (algorithm=="MH"){
      # Must be a symmetric proposal else MHstep is incorrect...
      beta.prop <- matrix(rmvnorm(n=1,mean=beta.t,sigma=k*V),ncol=1)
      mhs <- MHstep(beta.curr=beta.t,beta.prop=beta.prop,n=n,y=y,X=X,beta.0=beta.0,Sigma.0.inv=Sigma.0.inv,verbose=verbose)
      if (mhs$accept){
        beta.t <- beta.prop
        # Update counters:
        acc <- acc+1
        acc.since.last.tune <- acc.since.last.tune + 1
        if (verbose){cat("Accepted. :)\n")}
      } else {
        if (verbose){cat("Rejected. :(\n")}
      }
    } else if (algorithm=="MHwG"){
      # Metropolis within Gibbs:
      for (j in 1:p){
        # Must be a symmetric proposal else MHstep is incorrect...
        beta.prop <- beta.t
        beta.prop[j] <- beta.prop[j] + rnorm(n=1,mean=0,sd=k[j]*V[j])
        mhs <- MHstep(beta.curr=beta.t,beta.prop=beta.prop,n=n,y=y,X=X,beta.0=beta.0,Sigma.0.inv=Sigma.0.inv,verbose=verbose)
        if (mhs$accept){
          beta.t <- beta.prop
          # Update counters:
          acc[j] <- acc[j]+1
          acc.since.last.tune[j] <- acc.since.last.tune[j] + 1
          if (verbose){cat("Accepted. :)\n")}
        } else {
          if (verbose){cat("Rejected. :(\n")}
        }
      }
    }
    # Store the draw:
    if (iter>burnin){
      beta.samples[iter-burnin,] <- beta.t
    }
    # Update counter:
    iter.since.last.tune <- iter.since.last.tune+1    
    # Update the user:
    if (iter %% print.every == 0){
      if (algorithm=="MH"){
        cat(paste("Finished iteration ",iter,"... (acc rate = ",round(acc/iter,4),")\n",sep=""))
      } else {
        cat(paste("Finished iteration ",iter,"...\nacc rates:\n")) 
        print(round(acc/iter,4))        
      }
    }
  }
  return(beta.samples)
}

"MHstep" <- function(beta.curr,beta.prop,n,y,X,beta.0,Sigma.0.inv,verbose)
{
  p.curr <- blr.log.posterior(beta=beta.curr,n=n,y=y,X=X,beta.0=beta.0,Sigma.0.inv=Sigma.0.inv)
  p.prop <- blr.log.posterior(beta=beta.prop,n=n,y=y,X=X,beta.0=beta.0,Sigma.0.inv=Sigma.0.inv)
  # Metropolis-Hastings ratio:
  log.u <- log(runif(1))
  if (verbose){
    cat(paste("beta.curr:\n")) ; print(beta.curr)
    cat(paste("beta.prop:\n")) ; print(beta.prop)
    cat(paste("log(p(beta.curr)):\n")) ; print(p.curr)
    cat(paste("log(p(beta.prop)):\n")) ; print(p.prop)
    cat(paste("log(u) = ",log.u,"\n",sep=""))
  }
  return(list("accept"=as.logical(log.u <= p.prop - p.curr)))
}

