
##
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##

library(mvtnorm)
library(coda)

source("BLR_funcs.R")

#################################################
verbose <- FALSE
shortrun <- FALSE
niter  <- ifelse(shortrun,10000,500000)
burnin <- ifelse(shortrun,2000,10000)
retune <- ifelse(shortrun,200,500)
print.full.results <- FALSE
print.coefs <- TRUE
algorithm <- "MH"
standardize <- FALSE
print.every <- 5000
#################################################

# Read data:
dat <- read.table(file="breast_cancer.txt",header=TRUE)

# Extract X and y:
y <- as.numeric(dat$diagnosis)-1
n <- rep(1,nrow(dat))
X <- cbind(1,as.matrix(dat[,1:10]))
colnames(X)[1] <- "Intercept"
p <- ncol(X)
if (standardize){
  X[,2:p] <- apply(X[,2:p],2,function(x){(x-mean(x))/sd(x)})
}

rlr <- glm(y~0+X,family=binomial)
round(confint(rlr),4)

beta.0 <- matrix(rep(0,p))
Sigma.0.inv <- diag(rep(1/1000,p))
  
cat("Fitting Bayesian Logistic Regression...\n")

mcmctime <- system.time({
  blr <- bayes.logreg(n=n,y=y,X=X,beta.0=beta.0,Sigma.0.inv=Sigma.0.inv,
                    algorithm=algorithm,retune=retune,
                    print.every=print.every,
                    niter=niter,burnin=burnin,verbose=verbose)
})

cat(paste0("MCMC runtime: ",round(mcmctime["elapsed"],4),"\n")) 

# MHwG: 500 secs for 110k iters
# MH: 110 secs for 110k iters

resfilename <- paste0("bc_posteriors_",algorithm,"_",ifelse(standardize,"standardized",""),".txt")
sink(resfilename)
print(summary(blr))
cat("ACF's:\n")
print(apply(blr,2,function(x){acf(x,plot=FALSE)$acf[2]}))
cat("ESS's:\n")
print(effectiveSize(blr))
sink()

#coef(rlr) # MLE for comparison...

#pdf(paste0("bc_posteriors_",algorithm,"_",ifelse(standardize,"standardized",""),".pdf"))
png(paste0("bc_posteriors_",algorithm,"_",ifelse(standardize,"standardized","_%02d"),".png"))
plot(blr)
dev.off()

"rbppc" <- function(beta){
  a <- exp(X%*%matrix(beta))
  return(mean(rbinom(n=nrow(X),size=1,prob=a/(1+a))))
}

nppc <- 10000
ppc_beta_samples <- blr[sample(1:nrow(blr),nppc),]
ppc_check <- apply(ppc_beta_samples,1,rbppc)

pdf("bc_ppc.pdf")
truehist(ppc_check,nbins=20,xlab="Mean(y)",main="Posterior Predictive Distribution")
abline(v=mean(y),col="red",lwd=2.0)
dev.off()
