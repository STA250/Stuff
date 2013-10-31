#
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##

library(mvtnorm)
library(coda)

########################################################################################
########################################################################################
## Handle batch job arguments:

# 1-indexed version is used now.
args <- commandArgs(TRUE)

cat(paste0("Command-line arguments:\n"))
print(args)
cat('prey for me juicy')
####
# sim_start ==> Lowest simulation number to be analyzed by this particular batch job
###

#######################
sim_start <- 1000
length.datasets <- 200
#######################

if (length(args)==0){
  sinkit <- FALSE
  sim_num <- sim_start + 1
 # set.seed(131)
} else {
  # Sink output to file?
  sinkit <- TRUE
  # Decide on the job number, usually start at 1000:
  sim_num <- sim_start + as.numeric(args[1])
  # Set a different random seed for every job number!!!
  set.seed(762*sim_num + 1330991)
}

# Simulation datasets numbered 1001-1200
bayes.logreg=function(datas,beta.0,sd,niter=10000,burnin=1000,print.every=1000,retune=100,verbose=TRUE,paranum=2){
	#m is the whole trial and n is the time of win, x is paras
	cat('mission start','\n','your sd is ',sd,'\n');	
	paranum<-length(datas[1, ])-2;
	accept<-rep(0,paranum);  
	sd0<-sd
	accept1<-rep(0,paranum); 
	tnum=niter+burnin;
	ori=beta.0; 
	ray=matrix(0,nrow=tnum,ncol=paranum); #
	walk<-matrix(0,nrow=tnum,ncol=paranum)
	for (i in 1:length(ori)){
		walk[1:retune,i]<-rnorm(retune,0,sd[i])
	}   
      #cat(nrow(walk)); cat(ncol(walk));#
	uni=matrix(runif(tnum*paranum,0,1),nrow=tnum,ncol=paranum); #
	ray[1, ]=ori;
	burnin.tmp<-rep(0,paranum);
	for (j in 1:tnum){
		for (k in 1:paranum){
			star=ray[j,k]+walk[j,k];
			rays=ray[j, ]; 
			rays[k]=star;
			juicy<-calc(ray[j, ],rays,datas); #cat(j);cat('here');cat(k);cat('here');#
			#juicy<-juicy+log(dnorm(star,ori[k],sd[k])/dnorm(ray[j,k],ori[k],sd[k]))
			evil<-log(dnvnorm(star,beta.0[k],sd0[k]))-log(dnorm(ray[j,k],beta.0[k],sd0[k]))
			#cat(evil,'|')
			#cat(juicy,'|')
			juicy<-juicy+evil
			if(juicy>log(uni[j,k])){ray[j,k]=star; accept[k]<-accept[k]+1; burnin.tmp[k]<-burnin.tmp[k]+1; if(j>burnin){accept1[k]<-accept1[k]+1}};
          #cat('step');cat(j);cat(',');cat(k);cat(',finished');
		}
		if(j<tnum){ray[j+1, ]<-ray[j, ]}
		if(j<burnin+1 && j%%retune==0){
			bratio=burnin.tmp/retune;
			for (i in 1:length(bratio)){
				if(bratio[i]<0.1){sd[i]=sd[i]/4}	
				if(bratio[i]>=0.1 && bratio[i]<0.15){sd[i]=sd[i]/3.3}	
				if(bratio[i]>=0.15 && bratio[i]<0.2){sd[i]=sd[i]/2.8}	
				if(bratio[i]>=0.2 && bratio[i]<0.25){sd[i]=sd[i]/2}	
				if(bratio[i]>=0.25 && bratio[i]<0.27){sd[i]=sd[i]/1.5}	
				if(bratio[i]>=0.27 && bratio[i]<0.3){sd[i]=sd[i]/1.2}	
				if(bratio[i]>=0.55 && bratio[i]<0.6){sd[i]=sd[i]*1.1}	
				if(bratio[i]>=0.6 && bratio[i]<0.7){sd[i]=sd[i]*1.2}	
				if(bratio[i]>=0.7 && bratio[i]<0.75){sd[i]=sd[i]*1.5}	
				if(bratio[i]>=0.75 && bratio[i]<0.8){sd[i]=sd[i]*1.8}	
				if(bratio[i]>=0.8 && bratio[i]<0.85){sd[i]=sd[i]*2.5}	
				if(bratio[i]>=0.85 && bratio[i]<0.9){sd[i]=sd[i]*3.5}	
				if(bratio[i]>=0.9){sd[i]=sd[i]*4}
			}
			for (i in 1:length(ori)){
				walk[j+1:j+retune,i]<-rnorm(retune,0,sd[i])
			}		   
			cat('sd tuned at ',j,'  ',sd,'\n','the accept ratio is ',bratio,'\n')
			burnin.tmp<-rep(0,paranum);
		}	
		if(j==burnin){
			for (i in 1:length(ori)){
				walk[(j+1+retune):tnum,i]<-rnorm((niter-retune),0,sd[i])
			}
		}
		if(j%%1000==0){cat('the process till '); cat(j); cat(' is finished'); cat('\n')}
        #cat('\n');
	}
#	ratio<-accept/tnum; ratio1<-accept1/realrun; 
      #cat(accept); cat(' of '); cat(tnum); cat(' is accepted '); cat('\n');
#	cat('accept ratio of whole process is '); cat(ratio); cat('\n'); cat('accept ratio of realrun is '); cat(ratio1); cat('\n');
	return (ray[(burnin+1):tnum, ]);	
}
calc=function(now,star,datas,datat=1){
 	#rawdata<-datas[ ,1:(ncol(datas)-paranum)];
	#paras<-datas[ ,(ncol(datas)-paranum+1):(ncol(datas))]; #
	if(datat==1){      
		pnow=sum(apply(datas,1,function(x){m<-x[1]; n<-x[2]; da<-x[3:length(x)]; g<-exp(sum(now*da)); p<-(g/(1+g)); return(log(choose(n,m)*(p^m)*((1-p)^(n-m))))}));
		pstar=sum(apply(datas,1,function(x){m<-x[1]; n<-x[2]; da<-x[3:length(x)]; g<-exp(sum(star*da)); p<-(g/(1+g)); return(log(choose(n,m)*(p^m)*((1-p)^(n-m))))}));
        #cat (pnow);cat(pstar);
		return (pstar-pnow);
	}
}
#################################################
# Set up the specifications:
p<-2; #it seems that p is not set yetmaybe....
beta.0 <- matrix(c(0,0))
Sigma.0.inv <- diag(rep(1.0,p))
niter <- 10000
# etc... (more needed here)
#################################################

# Read data corresponding to appropriate sim_num:
filename<-paste0('data/blr_data_',sim_num,'.csv');
datas<-read.csv(filename,header=TRUE);
# Fit the Bayesian model:
out<-bayes.logreg(datas,rep(0,10),rep(1000,10),niter=1000000,burnin=10000,retune=400);
write.table(out,file='only.txt',sep=',',quote=FALSE,row.names=FALSE,col.names=FALSE);
# Extract posterior quantiles...
#p1<-out[ ,1]
#p2<-out[ ,2]
#q1<-quantile(p1,seq(.01,.99,.01))
#q2<-quantile(p2,seq(.01,.99,.01))
outfile<-paste0('results/blr_res_',sim_num,'.csv')
#write.table(data.frame("p1"=q1,"p2"=q2),file=outfile,sep=",",quote=FALSE,row.names=FALSE,col.names=FALSE)
cat('\n',mean(out[ ,1]),'\n',mean(out[ ,2]))
# Write results to a (99 x p) csv file...

# Go celebrate.
 
cat("done. :)\n")

