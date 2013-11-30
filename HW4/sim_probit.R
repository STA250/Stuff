
library(mvtnorm)

"sim_probit" <- function(m,p,beta.0,Sigma.0.inv)
{
  n <- 1
  X <- matrix(1,ncol=1,nrow=m)
  if (p>=2){
    for (i in 2:p){
      X <- cbind(X,rnorm(m))      
    }
  }
  beta <- matrix(rmvnorm(n=1,mean=beta.0,sigma=solve(Sigma.0.inv)),ncol=1)
  eta <- X%*%beta
  y <- rbinom(n=m,size=n,prob=pnorm(eta))
  return(list("y"=y,"X"=X,"n"=n,"beta"=beta))
}

"make_matrix" <- function(dat)
{
  ret <- cbind(dat$y,round(dat$X,3))
  colnames(ret) <- c("y",paste("X_",c(1:ncol(dat$X)),sep=""))
  return(ret)
}

set.seed(112312313)

n <- 100 
p <- 8
beta.0 <- rnorm(p)
Sigma.0.inv <- diag(1,p)

mini_data <- sim_probit(m=n,p=p,beta.0=beta.0,Sigma.0.inv=Sigma.0.inv)
out_dat <- make_matrix(dat=mini_data)
write.table(out_dat,file="mini_data.txt",col.names=TRUE,row.names=FALSE,quote=FALSE)
write.table(mini_data$beta,file="mini_pars.txt",col.names=TRUE,row.names=FALSE,quote=FALSE)

p <- 8
beta.0 <- rnorm(p)
Sigma.0.inv <- diag(1,p)

for (i in c(3:7)){
  data <- sim_probit(m=(10^i),p=p,beta.0=beta.0,Sigma.0.inv=Sigma.0.inv)
  out_dat <- make_matrix(dat=data)
  write.table(out_dat,file=sprintf("data_%02d.txt",i-2),col.names=TRUE,row.names=FALSE,quote=FALSE)
  write.table(data$beta,file=sprintf("pars_%02d.txt",i-2),col.names=TRUE,row.names=FALSE,quote=FALSE)
}








