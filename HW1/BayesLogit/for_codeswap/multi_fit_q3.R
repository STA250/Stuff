#I'm new
#
# Logistic regression
# 
# Y_{i} | \beta \sim \textrm{Bin}\left(n_{i},e^{x_{i}^{T}\beta}/(1+e^{x_{i}^{T}\beta})\right)
# \beta \sim N\left(\beta_{0},\Sigma_{0}\right)
#
##
cat('juicy with dream')
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
	#paranum=length(datas[1, ])-2;
	#m is the whole trial and n is the time of win, x is paras
	cat('mission start','\n','your sd is ',sd,'\n');	
	#accept<-rep(0,paranum);
	paranum<-length(datas[1, ])-2;  
	sd0<-sd
	accept1<-rep(0,paranum); 
	tnum=niter+burnin;
	ori=beta.0; 
	ray=matrix(0,nrow=tnum,ncol=paranum); #
	#walk<-matrix(0,nrow=tnum,ncol=paranum)
	burn.tmp<-0; 
      #cat(nrow(walk)); cat(ncol(walk));#
	uni=runif(tnum,0,1) #
	ray[1, ]=ori;
	burnin.tmp<-0;
	for (j in 1:tnum){
		now<-ray[j, ];
		star<-rmvnorm(1,now,diag(sd));
		juicy<-calc(now,star,datas);
		evil1<-log(dmvnorm(star,ori,diag(sd0)));
		evil2<-log(dmvnorm(now,ori,diag(sd0)));
		juicy<-juicy+evil1-evil2;
		if(juicy>log(uni[j])){
			ray[j, ]<-star; 
			burnin.tmp<-burnin.tmp+1;
		}
		if(j<burnin+1 && j%%retune==0){
			tmpratio<-burnin.tmp/retune;
			if(tmpratio>=0 && tmpratio<=0.1){sd<-sd/3};
			if(tmpratio>0.1 && tmpratio<=0.2){sd<-sd/2.5};
			if(tmpratio>.2 && tmpratio<=0.25){sd<-sd/2.2};
			if(tmpratio>0.25 && tmpratio<=0.28){sd<-sd/2};
			if(tmpratio>0.28 && tmpratio<=0.3){sd<-sd/1.7};
			if(tmpratio>0.3 && tmpratio<=0.35){sd<-sd/1.5};
			if(tmpratio>0.35 && tmpratio<=0.4){sd<-sd/1.3};
			if(tmpratio>0.4 && tmpratio<=0.45){sd<-sd/1.1};
			#if(tmpratio>0.45 && tmpratio<=0.5){sd<-sd/1.1};
			#if(tmpratio>0.56 && tmpratio<=0.6){sd<-sd*1.15};
			if(tmpratio>0.6 && tmpratio<=0.65){sd<-sd*1.25};
			if(tmpratio>0.65 && tmpratio<=0.68){sd<-sd*1.4};
			if(tmpratio>0.68 && tmpratio<=0.75){sd<-sd*1.9};
			if(tmpratio>0.75 && tmpratio<=0.8){sd<-sd*2.7};
			if(tmpratio>0.8 && tmpratio<=0.9){sd<-sd*3.4};
			if(tmpratio>0.9 && tmpratio<=1){sd<-sd*4};
			cat('sd retuned till ',j,'\n',sd,' with ratio as ',tmpratio,'\n');
			burnin.tmp<-0;
		}
		if(j%%1000==0){cat('the process till '); cat(j); cat(' is finished'); cat('\n')}
		if(j<tnum){ray[j+1, ]<-ray[j, ]}
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
		#pnow=apply(datas,1,function(x){da<-x[3:length(x)];g<-exp(sum(now*da));p<-(g/(1+g));return(p)});
		#pstar=apply(datas,1,function(x){da<-x[3:length(x)]; g<-exp(sum(star*da));p<-(g/(1+g));return(p)});
		#pnew=log(pstar)-log(pnow);
		#pnew1=log(1-pstar)-log(1-pnow);
		#dan<-cbind(datas[ ,1:2],pnew,pnew1);
		#juicy<-sum(apply(dan,1,function(x){m<-x[2];n<-x[1];return(m*x[3]+(m-n)*x[4])}));
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
#filename<-paste0('data/blr_data_',sim_num,'.csv');
filename<-'cans.txt'
#datas<-read.csv(filename,header=TRUE);
datas<-read.table(filename,header=FALSE);
datas[3:12]<-scale(datas[ ,3:12])/10000
# Fit the Bayesian model:
out<-bayes.logreg(datas,rep(0,10),rep(1000,10),niter=70000,burnin=12000,retune=400)
plot(out[ ,1]);
plot(out[ ,2]);
plot(out[ ,3]);
plot(out[ ,4]);
# Extract posterior quantiles...
#p1<-out[ ,1]
#p2<-out[ ,2]
#q1<-quantile(p1,seq(.01,.99,.01))
#q2<-quantile(p2,seq(.01,.99,.01))
mean<-apply(out,2,mean); cat(mean,'\n');
#outfile<-paste0('results/blr_res_',sim_num,'.csv')
outfile<-'juicylove.txt'; 
#write.table(data.frame("p1"=q1,"p2"=q2),file=outfile,sep=",",quote=FALSE,row.names=FALSE,col.names=FALSE)
write.table(out,file=outfile,sep=",",quote=FALSE,row.names=FALSE,col.names=FALSE)
#cat('\n',mean(out[ ,1]),'\n',mean(out[ ,2]))
# Write results to a (99 x p) csv file...
# Go celebrate.
 
cat("done. :)\n")

