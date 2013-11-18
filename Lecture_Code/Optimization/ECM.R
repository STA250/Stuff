# Pre-reqs:
#require(devtools)
#install_github(username="pdbaines",repo="EM")

library(EM)

"ECM.update" <- function(theta.t,y.obs,fixed,verbose)
{
  # ECM update:
  alpha.t <- theta.t[1]
  beta.t <- theta.t[2]
  n <- fixed$ncom
  nmis <- y.obs$nmis
  yobs <- y.obs$y
  
  # Update beta:
  beta.t1 <- (n*alpha.t)/(fixed$yobssum + nmis*(alpha.t/beta.t))
  
  # NR for alpha:
  #NewtonRaphson(func,deriv.func,init,tol=1e-10,maxit=1000,simplify=TRUE,...)
  "afunc" <- function(alpha,beta,theta.t,n,yobs,nmis){
    return(n*(log(beta)-digamma(alpha)) + sum(log(yobs)) +
      nmis*(digamma(theta.t[1])-log(theta.t[2])))
  }
  "afunc.deriv" <- function(alpha,beta,theta.t,n,yobs,nmis)
  {
    return(-n*psigamma(alpha,1))
  }
  
  alpha.t1 <- NewtonRaphson(func=afunc,deriv.func=afunc.deriv,init=alpha.t,
                            beta=beta.t1,theta.t=theta.t,n=n,yobs=yobs,nmis=nmis)
  
  if (verbose){
    cat("\n===========\n")
    cat(paste0("theta.t  = (",alpha.t,",",beta.t,")\n"))
    cat(paste0("theta.t1 = (",alpha.t1,",",beta.t1,")\n"))
    cat("===========\n")
  }
  return(c(alpha.t1,beta.t1))
}

alpha <- 1.0
beta <- 0.5
nobs <- 100
nmis <- 50

yobs <- rgamma(n=nobs,alpha,beta)
ymis <- rgamma(n=nmis,alpha,beta)
ycom <- c(yobs,ymis)

y.obs <- list("y"=yobs,"nobs"=nobs,"nmis"=nmis)
fixed <- list("yobssum"=sum(yobs),"ncom"=nmis+nobs)
theta.0 <- c(1,1)
max.iter <- 100
verbose <- TRUE

gamma.ecm <- EM(y.obs=y.obs,theta.0=theta.0,fixed=fixed,
                update=ECM.update,
                max.iter=max.iter,verbose=verbose)

par(mfrow=c(1,1))
matplot(gamma.ecm$paths$theta,type="l",xlab="iteration",ylab="theta",
        main="ECM Paths: Incomplete Gamma Model")

# Can compute observed data log-likelihood here directly (missing data integrates out):
"gamma.llike" <- function(par,y.obs)
{
  return(-sum(dgamma(y.obs$y,par[1],par[2],log=TRUE)))
}

gamma.optim <- optim(par=theta.0,fn=gamma.llike,y.obs=y.obs)

cat("Direct maximization:\n") ; print(gamma.optim$par)
cat("Using EM:\n") ; print(gamma.ecm$theta)


