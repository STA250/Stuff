# Latent variables:

invlogit  <- function(x){exp(x)/(1+exp(x))}
invprobit <- function(x){pnorm(x)}

beta <- c(0.5,1.2)
sigma <- 1

X <- cbind("(Intercept)"=1,"x"=rnorm(100))
head(X)

Z <- X%*%beta + matrix(rnorm(n=nrow(X),sd=sigma),ncol=1)

# The EM Algorithm for Probit Regression:

"rtruncnorm" <- function(n,mean,sd,lo=-Inf,hi=Inf){
  u <- runif(n=n,min=pnorm(lo,mean=mean,sd=sd),max=pnorm(hi,mean=mean,sd=sd))
  return(qnorm(u,mean=mean,sd=sd))
}

library(MASS)
par(mfrow=c(2,2))
x1 <- rtruncnorm(n=50000,mean=1,sd=1,lo=0,hi=2)
x2 <- rtruncnorm(n=50000,mean=1,sd=1,lo=-Inf,hi=2)
x3 <- rtruncnorm(n=50000,mean=1,sd=1,lo=1,hi=Inf)
x4 <- rtruncnorm(n=50000,mean=1,sd=1,lo=-1,hi=1)
truehist(x1,nbins=20)
truehist(x2,nbins=20)
truehist(x3,nbins=20)
truehist(x4,nbins=20)

"Etruncnorm" <- function(mean,sd,lo,hi)
{
  return(mean + sd*(dnorm((lo-mean)/sd)-dnorm((hi-mean)/sd))/(pnorm((hi-mean)/sd)-pnorm((lo-mean)/sd)))
}

c("a"=mean(x1),"e"=Etruncnorm(mean=1,sd=1,lo=0,hi=2))
c("a"=mean(x2),"e"=Etruncnorm(mean=1,sd=1,lo=-Inf,hi=2))
c("a"=mean(x3),"e"=Etruncnorm(mean=1,sd=1,lo=1,hi=Inf))
c("a"=mean(x4),"e"=Etruncnorm(mean=1,sd=1,lo=-1,hi=1))

"probit_EM" <- function(formula, data, weights, subset, 
                      na.action, start = NULL, etastart, mustart,
                      model = TRUE, x = FALSE, y = TRUE, contrasts = NULL,
                      tol=1e-10, max_iter=100, verbose=FALSE,
                      MCEM=FALSE, n_mc=1000, print_every=10,debug=FALSE,
                      store_path=FALSE, ...) 
{
  call <- match.call()
  if (missing(data)) 
    data <- environment(formula)
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "weights", "na.action", 
               "etastart", "mustart", "offset"), names(mf), 0)
  mf <- mf[c(1, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  Y <- model.response(mf, "any")
  if (length(dim(Y)) == 1) {
    nm <- rownames(Y)
    dim(Y) <- NULL
    if (!is.null(nm)) 
      names(Y) <- nm
  }
  X <- if (!is.empty.model(mt)) 
    model.matrix(mt, mf, contrasts)
  else matrix(, NROW(Y), 0)
  weights <- as.vector(model.weights(mf))
  if (!is.null(weights) && !is.numeric(weights)) 
    stop("'weights' must be a numeric vector")
  offset <- as.vector(model.offset(mf))
  if (!is.null(weights) && any(weights < 0)) 
    stop("negative weights not allowed")
  if (!is.null(offset)) {
    if (length(offset) == 1) 
      offset <- rep(offset, NROW(Y))
    else if (length(offset) != NROW(Y)) 
      stop(gettextf("number of offsets is %d should equal %d (number of observations)", 
                    length(offset), NROW(Y)), domain = NA)
  }
  mustart <- model.extract(mf, "mustart")
  etastart <- model.extract(mf, "etastart")
  
  converged <- function(new,old,tol){
    if (sum(((old-new)/new)^2)<tol){
      return(TRUE)
    } else {
      return(FALSE)
    }
  }
  E_step <- function(X,Y,beta_current,MCEM=FALSE,verbose=FALSE,debug=FALSE){
    Z_new <- X%*%beta_current
    if (debug){
      MCEM_original <- MCEM
      MCEM <- FALSE
    }
    if (!MCEM){
      # Analytical:
      lob <- ifelse(Y==0,-Inf,0)
      hib <- ifelse(Y==0,0,Inf)
      Z_t1 <- Etruncnorm(mean=Z_new,sd=1,lo=lob,hi=hib)
    }
    if (debug){
      Z_analytical <- Z_t1
      MCEM <- TRUE
    }
    if (MCEM){
      # MCEM:
      Z_t1 <- rep(NA,length(Z_new))
      lob <- ifelse(Y==0,-Inf,0)
      hib <- ifelse(Y==0,0,Inf)
      for (i in 1:length(Z_new)){
        Z_t1[i] <- mean(rtruncnorm(n=n_mc,mean=Z_new[i],sd=1,lo=lob[i],hi=hib[i]))
      }
    }
    if (debug){
      Z_approx <- Z_t1
      if (MCEM_original){
        Z_t1 <- Z_approx
      } else {
        Z_t1 <- Z_analytical
      }
      cat("Analytical vs. Approximate values for Z^{(t+1)}:\n")
      print(data.frame("analytical"=Z_analytical,"approx"=Z_approx))
    }
    return(Z_t1)
  }
  obs_loglike <- function(X,Y,beta,verbose=FALSE){
    phats <- pnorm(X%*%matrix(beta,ncol=1))
    return(sum(ifelse(Y==1,log(phats),log(1-phats))))
  }
  
  # Starting states:
  beta_new     <- solve(t(X)%*%X)%*%t(X)%*%matrix(Y,ncol=1)
  beta_current <- matrix(Inf,nrow=ncol(X),ncol=1)
  iter <- 0
  beta_path <- NULL
  Z_path    <- NULL
  ll_path   <- NULL
  
  while (!converged(beta_new,beta_current,tol) && iter<max_iter){
    beta_current <- beta_new
    Z <- E_step(X,Y,beta_current,MCEM=MCEM,debug=debug,verbose=verbose)
    beta_new <- solve(t(X)%*%X)%*%t(X)%*%matrix(Z,ncol=1)
    iter <- iter+1
    if (verbose){
      cat(paste("beta_{",iter,"}:\n",sep=""))
      print(beta_new)
    }
    if (iter%%print_every == 0){
      cat(paste0("Finished iteration ",iter,"...\n"))
    }
    if (store_path){
      beta_path <- rbind(beta_path,matrix(beta_new,nrow=1))
      Z_path    <- rbind(Z_path,matrix(Z,nrow=1))
      ll_path   <- c(ll_path,obs_loglike(X,Y,beta_new,verbose=verbose))
    }
  }
  rlist <- list("beta"=beta_new,
                "beta_path"=beta_path,
                "Z_path"=Z_path,
                "ll_path"=ll_path,
                "Xmatrix"=X)
  return(rlist)
}

library(MASS)
birthwt.glm <- glm(low~age,data=birthwt,family=binomial(link="probit"))
summary(birthwt.glm)

# Regular EM:
cat("Running regular EM...\n")
bwt_em <- probit_EM(low~age,data=birthwt)
cat("Running regular EM... (storing paths)\n")
wpaths <- probit_EM(low~age,data=birthwt,store_path=TRUE)

# MCEM:
n_mc <- 10000
debug <- FALSE # To view difference b/w approx and exact E-step
cat("Running MCEM...\n")
bwt_mcem <- probit_EM(low~age,data=birthwt,MCEM=TRUE,n_mc=n_mc,debug=debug)
cat("Running MCEM... (storing paths)\n")
mcwpaths <- probit_EM(low~age,data=birthwt,MCEM=TRUE,n_mc=n_mc,debug=debug,store_path=TRUE)
  
t(bwt_em$beta)
coef(birthwt.glm)

par(mfrow=c(1,2))
with(wpaths,{
  xlims <- range(beta_path[,1])
  ylims <- range(beta_path[,2])
  plot(beta_path,type='n',xlim=xlims,ylim=ylims,
       xlab="beta_0",ylab="beta_1",
       main="Path of the EM-Algorithm (betas)")
  colvec <- rev(heat.colors(nrow(beta_path)+4))
  colvec <- colvec[5:length(colvec)]
  for (i in 1:nrow(beta_path)){
    Sys.sleep(0.1)
    if (i>1)
      arrows(x0=beta_path[i-1,1],
             y0=beta_path[i-1,2],
             x1=beta_path[i,1],
             y1=beta_path[i,2],
             col=colvec[i-1],lwd=1.4)
    points(list("x"=beta_path[i,1],
                "y"=beta_path[i,2]),
           cex=2,pch=16,col=colvec[i])
  }
  # beta_path,pch=as.character(1:nrow(beta_path))))
})

with(mcwpaths,{
  xlims <- range(beta_path[,1])
  ylims <- range(beta_path[,2])
  plot(beta_path,type='n',xlim=xlims,ylim=ylims,
       xlab="beta_0",ylab="beta_1",
       main="Path of the MCEM-Algorithm (betas)")
  colvec <- rev(heat.colors(nrow(beta_path)+4))
  colvec <- colvec[5:length(colvec)]
  for (i in 1:nrow(beta_path)){
    Sys.sleep(0.1)
    if (i>1)
      arrows(x0=beta_path[i-1,1],
             y0=beta_path[i-1,2],
             x1=beta_path[i,1],
             y1=beta_path[i,2],
             col=colvec[i-1],lwd=1.4)
    points(list("x"=beta_path[i,1],
                "y"=beta_path[i,2]),
           cex=2,pch=16,col=colvec[i])
  }
  # beta_path,pch=as.character(1:nrow(beta_path))))
})

birthtable <- table(birthwt$low,birthwt$age)
ages <- dim(birthtable)[2]
# Need to get rid of the 'phantom' 0's:
birthtable[birthtable == 0] <- NA

par(mfrow=c(1,2))
with(wpaths,{
  plot(ll_path,type='n',
       xlab="Iteration",ylab="Observed Data Log-Likelihood",
       main="Path of the EM-Algorithm (obs-log-like)")
  abline(h=0,lwd=2)
  colvec <- rev(rainbow(length(ll_path)+4))
  colvec <- colvec[5:length(colvec)]
  for (i in 1:length(ll_path)){
    Sys.sleep(0.1)
    points(list("x"=i,
                "y"=ll_path[i]),
           cex=1.2,pch=16,col=colvec[i])
    if (i>1)
      arrows(x0=i-1,
             y0=ll_path[i-1],
             x1=i,
             y1=ll_path[i],
             col=colvec[i],lwd=1.4)    
  }
})

with(mcwpaths,{
  plot(ll_path,type='n',
       xlab="Iteration",ylab="Observed Data Log-Likelihood",
       main="Path of the MCEM-Algorithm (obs-log-like)")
  abline(h=0,lwd=2)
  colvec <- rev(rainbow(length(ll_path)+4))
  colvec <- colvec[5:length(colvec)]
  for (i in 1:length(ll_path)){
    Sys.sleep(0.1)
    points(list("x"=i,
                "y"=ll_path[i]),
           cex=1.2,pch=16,col=colvec[i])
    if (i>1)
      arrows(x0=i-1,
             y0=ll_path[i-1],
             x1=i,
             y1=ll_path[i],
             col=colvec[i],lwd=1.4)    
  }
})

#########################################

par(mfrow=c(1,2))
plot(mcwpaths$beta_path[,1],type="l",xlab="beta_0")
plot(mcwpaths$beta_path[,2],type="l",ylab="beta_1")

#rm(wpaths)
#rm(mcwpaths)
