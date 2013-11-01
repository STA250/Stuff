
# Make 2D Histogram:

#res <- read.table("mini_result.txt",sep=",",header=F)
res <- read.table("full_results.txt",sep=",",header=F)
head(res)

x <- apply(res[,1:2],1,mean)
y <- apply(res[,3:4],1,mean)
z <- res[,5]

colvec <- terrain.colors(max(res[,5])+1)
mf <- 4
px <- 480*mf
png("hist2d.png",width=px,height=px)
plot(range(x),range(y),type="n",
     main="Bivariate Histogram: STA 250 HW2",
     xlab="x",ylab="y",
     cex.main=mf,cex.lab=mf,cex.axis=mf)
for (i in 1:nrow(res)){
  polygon(x=c(res[i,1],res[i,2],res[i,2],res[i,1]),
          y=c(res[i,3],res[i,3],res[i,4],res[i,4]),
          col=colvec[res[i,5]])
  if (i%%10000==0){
    cat(paste0("Finished row ",i,"\n"))
  }
}
dev.off()
