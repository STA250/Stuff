
rm(list=ls())

library(MASS)
data(newcomb)

do.plot <- TRUE

y <- newcomb
truehist(y)

n <- length(y)
ybar <- mean(y)
s.sq <- var(y)

mu.0 <- 0
sig.0.sq <- 10^2
kappa.0 <- 0.0 # Improper, flat prior on mu
nu.0 <- 0.0 # Improper 1/sigma^2 prior on sigma^2

kappa.n <- kappa.0 + n
nu.n <- nu.0 + n
sig.n.sq <- (nu.0*sig.0.sq + (n-1)*s.sq + (kappa.0*n*(ybar-mu.0)^2)/kappa.n)/nu.n
mu.n <- (kappa.0*mu.0 + n*ybar)/kappa.n

library(geoR)

m <- 20000

sig.sq.samples <- rinvchisq(n=m,df=nu.n,scale=sig.n.sq)
mu.samples <- rnorm(n=m,mean=mu.n,sd=sqrt(sig.sq.samples/kappa.n))

library(ggplot2)

"gghist2d" <- function(xobj,yobj,xname="x",yname="y",xlim=NULL,ylim=NULL,points=TRUE,
                       mytitle="Bivariate Density Plot",
                       s.point=NULL,s.col=NULL,s.shape=NULL,s.size=NULL,
                       verbose=TRUE)
{
    df_command <- paste(".tmp.df <- data.frame(",xname,"=",xobj,",",yname,"=",yobj,")",sep="")
    plot_command <- paste("m <- ggplot(.tmp.df,aes(",xname,",",yname,")) ",
      ifelse(points,"+ geom_point() ",""),
      ifelse(!is.null(xlim),paste(" + scale_x_continuous(limits=",xlim,")",sep=""),""),
      ifelse(!is.null(ylim),paste(" + scale_y_continuous(limits=",ylim,")",sep=""),""),
      " ; print(m + geom_density2d()"," + opts(title='",mytitle,"')",
      ifelse(!is.null(s.point),paste(" + geom_point(",s.point,",col='",
          ifelse(!is.null(s.col),paste(s.col,"'",sep=""),"'red'"),",shape=",
          ifelse(!is.null(s.shape),paste(s.shape,sep=""),"18"),",size=",
          ifelse(!is.null(s.size),paste(s.size,sep=""),"4"),
                                                  ")",sep=""),""),  ")",sep="")

    
    #if (verbose)  cat("Creating temporary data.frame...\n")
    #eval(parse(text=df_command))
    #if (verbose)  cat("Creating 2-D density plot...\n")
    #eval(parse(text=plot_command))

    all_commands <- paste(df_command," ; ",plot_command,sep="")
    eval(parse(text=all_commands))
    return(all_commands)
}

if (do.plot){
	gghist2d(xobj="mu.samples",yobj="sqrt(sig.sq.samples)",xname="mu",yname="sigma")
}

quantile(mu.samples,prob=c(0.025,0.50,0.975))

# Posterior predictive datasets:

ppc.datasets <- matrix(nrow=m,ncol=n)
rownames(ppc.datasets) <- paste("PPC_Dataset_",1:m,sep="")

for (i in 1:m){
  ppc.datasets[i,] <- rnorm(n=n,mean=mu.samples[i],sd=sqrt(sig.sq.samples[i]))
}

head(ppc.datasets)

par(mfrow=c(3,3))
xlim <- c(min(min(y),min(ppc.datasets[1:8,])),max(max(newcomb),max(ppc.datasets[1:8,])))
truehist(y,xlim=xlim,main="Real Data",col="red")
for (i in 1:8){
  truehist(ppc.datasets[i,],xlim=xlim,main=paste("Posterior Predictive Dataset ",i,sep=""))
}
dev.off()

# Posterior predictive test statistics:
nstats <- 4
ppc.stats.ref <- matrix(NA,nrow=m,ncol=nstats)
colnames(ppc.stats.ref) <- c("min","max","mean","median")

for (i in 1:m){
  ppc.stats.ref[i,1] <- min(ppc.datasets[i,])
  ppc.stats.ref[i,2] <- max(ppc.datasets[i,])
  ppc.stats.ref[i,3] <- mean(ppc.datasets[i,])
  ppc.stats.ref[i,4] <- median(ppc.datasets[i,])
}

ppc.stats.obs <- c("min"=min(y),"max"=max(y),"mean"=mean(y),"median"=median(y))

par(mfrow=c(2,2))
for (i in 1:nstats){
  xlim <- c(min(c(ppc.stats.ref[,i],ppc.stats.obs[i])),max(c(ppc.stats.ref[,i],ppc.stats.obs[i])))
  truehist(ppc.stats.ref[,i],xlim=xlim,nbins=100,
    main=paste("Posterior Predictive Distribution of ",colnames(ppc.stats.ref)[i],sep=""),
    xlab=colnames(ppc.stats.ref)[i])
  abline(v=ppc.stats.obs[i],col="red",lwd=2.0)
}


"switches" <- function(x){
  return(sum(abs(diff(x))))
}

x <- c(1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0)
n <- length(x)

n.successes <- sum(x)
n.failures <- sum(1-x)

prior.alpha <- 0.5
prior.beta <- 0.5

post.alpha <- prior.alpha + n.successes
post.beta <- prior.beta + n.failures

m <- 10000

theta.samples <- rbeta(n=m,post.alpha,post.beta)

# Posterior predictive datasets:

ppc.datasets <- matrix(nrow=m,ncol=n)
rownames(ppc.datasets) <- paste("PPC_Dataset_",1:m,sep="")

for (i in 1:m){
  ppc.datasets[i,] <- rbinom(n=n,size=1,prob=theta.samples[i])
}

head(ppc.datasets)

