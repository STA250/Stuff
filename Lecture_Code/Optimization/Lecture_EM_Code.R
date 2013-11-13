
# Latent variables:

invlogit  <- function(x){exp(x)/(1+exp(x))}
invprobit <- function(x){pnorm(x)}

beta <- c(0.5,1.2)
sigma <- 1

X <- cbind("(Intercept)"=1,"x"=rnorm(100))
head(X)

Z <- X%*%beta + matrix(rnorm(n=nrow(X),sd=sigma),ncol=1)

pdf("figure1.pdf")
par(mfrow=c(1,2))
plot(y=Z,x=X[,"x"],
     xlab="x",ylab="z",pch=16,
     col=ifelse(Z>0,"red","blue"),
     main="Probit Regression: Latent Variable")
abline(h=0,lwd=1.8,lty=3)
abline(a=beta[1],b=beta[2],lwd=1.8,lty=2)
legend(-2.0,3.0,legend=c("True Regression Line","Lethal Dosage Threshold"),
       lty=c(2,3),lwd=1.8,cex=0.6)

Y <- ifelse(Z>0,1,0)
plot(y=Y,x=X[,"x"],
     xlab="x",ylab="y",pch=16,
     col=ifelse(Y,"red","blue"),
     main="Observed Binary Data")
dev.off()

jitter.binary <- function(a, jitt=.05){
  ifelse(a==0, runif(length(a),0,jitt),runif(length(a),1-jitt,1))
}

pdf("figure1b.pdf")
par(mfrow=c(1,2))
plot(y=Z,x=X[,"x"],
     xlab="x",ylab="z",pch=16,
     col=ifelse(Z>0,"red","blue"),
     main="Probit Regression: Latent Variable")
abline(h=0,lwd=1.8,lty=3)
abline(a=beta[1],b=beta[2],lwd=1.8,lty=2)
legend(-2.0,3.0,legend=c("True Regression Line","Lethal Dosage Threshold"),
       lty=c(2,3),lwd=1.8)

Y <- ifelse(Z>0,1,0)
plot(y=jitter.binary(Y),x=X[,"x"],
     xlab="x",ylab="y",pch=16,cex=0.8,
     col=ifelse(Y,"red","blue"),
     main="Observed Binary Data (jittered)")
dev.off()

x <- rep(seq(-5,5,by=0.1),each=100)
X.2 <- cbind("(Intercept)"=1,"x"=x)
head(X.2)

Z.2 <- X.2%*%beta + matrix(rnorm(n=nrow(X.2),sd=sigma),ncol=1)
Y.2 <- ifelse(Z.2>0,1,0)

pdf("figure2.pdf")
par(mfrow=c(1,2))
plot(y=Z.2,x=X.2[,"x"],
     xlab="x",ylab="z",pch=16,cex=0.3,
     col=ifelse(Z.2>0,"red","blue"),
     main="Probit Regression: Latent Variable")
abline(h=0,lwd=1.8,lty=3)
abline(a=beta[1],b=beta[2],lwd=1.8,lty=2)
legend(-4.8,9.5,legend=c("True Regression Line","Lethal Dosage Threshold"),
       lty=c(2,3),lwd=1.8)

phat.2 <- NULL
for (i in 1:length(unique(x)))
  phat.2 <- c(phat.2,mean(Z.2[as.numeric(x)==unique(x)[i],] > 0))

tdf <- data.frame("Y"=Y.2,"x"=X.2[,"x"])
head(tdf)

plot(y=phat.2,x=unique(x),pch=16,cex=0.8,
     xlab="x",ylab="Proportion of Y==1",
     main="Proportions of 1's and 0's vs. x")
curve(invprobit(beta[1]+beta[2]*x),col="red",add=TRUE)
dev.off()


# The EM Algorithm for Probit Regression:

probit_EM <- function(formula, data, weights, subset, 
    na.action, start = NULL, etastart, mustart,
    model = TRUE, x = FALSE, y = TRUE, contrasts = NULL,
    tol=1e-10, max_iter=100, verbose=FALSE,
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
    E_step <- function(X,Y,beta_current,verbose=FALSE){
      Z_new <- X%*%beta_current
      num_Z_new <- dnorm(Z_new)*ifelse(Y==1,+1,-1)
      den_Z_new <- ifelse(Y==1,1-pnorm(-Z_new),pnorm(-Z_new))
      Z_new <- Z_new + (num_Z_new/den_Z_new)
      return(Z_new)
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
      Z <- E_step(X,Y,beta_current,verbose=verbose)
      beta_new <- solve(t(X)%*%X)%*%t(X)%*%matrix(Z,ncol=1)
      iter <- iter+1
      if (verbose){
        cat(paste("beta_{",iter,"}:\n",sep=""))
        print(beta_new)
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

bwt_em <- probit_EM(low~age,data=birthwt)

wpaths <- probit_EM(low~age,data=birthwt,store_path=TRUE)

t(bwt_em$beta)
coef(birthwt.glm)

with(wpaths,{
  xlims <- range(beta_path[,1])
  ylims <- range(beta_path[,2])
  plot(beta_path,type='n',xlim=xlims,ylim=ylims,
       xlab="beta_0",ylab="beta_1",
       main="Path of the EM-Algorithm (betas)")
  colvec <- rev(heat.colors(nrow(beta_path)+4))
  colvec <- colvec[5:length(colvec)]
  for (i in 1:nrow(beta_path)){
    Sys.sleep(1)
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

wpaths$Z_path[,130]

with(wpaths,{
  xlims <- range(Xmatrix[,2])
  ylims <- range(Z_path)  
  plot(Z_path[1,],type='n',xlim=xlims,ylim=ylims,
       xlab="Maternal Age",ylab="Z",
       main="Path of the EM-Algorithm (Z)")
  abline(h=0,lwd=2)
  colvec <- rev(rainbow(nrow(Z_path)+4))
  colvec <- colvec[5:length(colvec)]
  for (i in 1:nrow(Z_path)){
    Sys.sleep(1)
    points(list("x"=Xmatrix[,2],
                "y"=Z_path[i,]),
           cex=1.2,pch=16,col=colvec[i])
  }
         # beta_path,pch=as.character(1:nrow(beta_path))))
})


with(wpaths,{
  plot(ll_path,type='n',
       xlab="Iteration",ylab="Observed Data Log-Likelihood",
       main="Path of the EM-Algorithm (obs-log-like)")
  abline(h=0,lwd=2)
  colvec <- rev(rainbow(length(ll_path)+4))
  colvec <- colvec[5:length(colvec)]
  for (i in 1:length(ll_path)){
    Sys.sleep(1)
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



rm(wpaths)

