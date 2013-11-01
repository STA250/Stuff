
# Read in and process BLB results:

mini <- FALSE
if (mini){
	d <- 40
} else {
	d <- 1000
}

# BLB specs:
s <- 5 # 50
r <- 50 # 100

outpath <- "output"
respath <- "final"

if (mini){
	rootfilename <- "blb_lin_reg_mini"
} else {
	rootfilename <- "blb_lin_reg_data"
}

results.se.filename <- paste0(respath,"/",rootfilename,"_s",s,"_r",r,"_SE.txt")
results.est.filename <- paste0(respath,"/",rootfilename,"_s",s,"_r",r,"_est.txt")

outfile <- function(outpath,r_index,s_index){
	return(paste0(outpath,"/","coef_",sprintf("%02d",s_index),"_",sprintf("%02d",r_index),".txt"))
}

coefs <- vector("list",s)
blb_est <- blb_se <- matrix(NA,nrow=s,ncol=d)

# Compute BLB SE's:
for (s_index in 1:s){
	coefs[[s_index]] <- matrix(NA,nrow=r,ncol=d)
	for (r_index in 1:r){
		tmp.filename <- outfile(outpath,r_index,s_index)
		tryread <- try({tmp <- read.table(tmp.filename,header=TRUE)},silent=TRUE)
		if (class(tryread)=="try-error"){
			errmsg <- paste0("Failed to read file: ",tmp.filename)
			stop(errmsg)
		}
		if (nrow(tmp) != d){
			stop(paste0("Incorrect number of rows in: ",tmp.filename))
		}
		coefs[[s_index]][r_index,] <- as.numeric(tmp[,1])
	}
	blb_est[s_index,] <- apply(coefs[[s_index]],2,mean)
	# SD for each subsample:
	blb_se[s_index,] <- apply(coefs[[s_index]],2,sd)
}

# Average over subsamples:
blb_final_est <- apply(blb_est,2,mean)
blb_final_se <- apply(blb_se,2,mean)

cat("Experimental Final BLB Estimates's (Note: These are biased in general):\n")
print(blb_final_est)

cat("Final BLB SE's:\n")
print(blb_final_se)

cat("Writing to file...\n")
write.table(file=results.se.filename,blb_final_se,row.names=F,quote=F)
#write.table(file=results.est.filename,blb_final_est,row.names=F,quote=F)
cat("done. :)\n")





