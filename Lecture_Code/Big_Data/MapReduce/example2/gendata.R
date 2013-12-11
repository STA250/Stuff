
setwd("~/Dropbox/Documents/Davis_Teaching/Stat_250/Fall_2013/GitHub/Stuff/Lecture_Code/Big_Data/example2/")

mini <- TRUE

if (mini){
  J <- 100
  n <- 1000  
} else {
  J <- 1000
  n <- 10000000
}

cat("Generating data...\n")

# ~2 secs on my ancient MBP

print(system.time({
x <- sample(1:J,size=n,replace=TRUE) ;
mu <- rt(n=J,df=1) ;
y <- rnorm(n=n,mean=mu[x],sd=1.0)
}))

cat("Writing data to file...\n")

# ~2 mins on my ancient MBP

outfile <- ifelse(mini,"mini_groups.txt","groups.txt")
print(system.time({write.table(file=outfile,
            data.frame("x"=x,"y"=y),
            col.names=F,row.names=F,sep="\t")}))

# ~217MB file produced.

if (mini){
  foo <- read.table("mini_groups.txt",header=F)
  colnames(foo) <- c("x","y")
  for (i in 1:100){
    cat(i,": ",mean(y[x==i]),"\n",sep="")
  }
}
