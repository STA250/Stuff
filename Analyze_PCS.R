
setwd("~/Dropbox/Documents/Davis_Teaching/Stat_250/Fall_2013/GitHub/STA250_Secret/Lectures/")
#setwd("~/Dropbox/Documents/Davis_Teaching/Stat_250/Fall_2013/Admin/")

pcs <- read.csv("Pre_Course_Survey.csv",header=TRUE)
head(pcs)

library(MASS)
library(ggplot2)
library(xtable)

"count_responses" <- function(x,cats,do_sort=TRUE,decreasing=TRUE,...)
{
  # Check categories...
  k <- length(cats)
  if (!(k>0) || class(cats)!="character"){
    stop("'cats' must be a vector of strings")
  }
  # Create return vector
  ret <- rep(NA,k)
  names(ret) <- cats
  # Grep for responses:
  for (i in 1:length(cats)){
    ret[i] <- length(grep(cats[i],x,...))
  }
  if (do_sort){
    ret <- sort(ret,decreasing=decreasing)
  }
  return(ret)
}

courses <- c("STA 106","STA 108","STA 131A","STA 131BC","STA 135","STA 137","STA 141","STA 145","STA 231ABC","STA 242","STA 243")
topics <- c("Maximum Likelihood Estimation","Bayesian Inference","MCMC","The EM Algorithm","Mixed Effects Models","Random Forests","Logistic Regression","Time Series Analysis","The Bootstrap") 
compies <- c("Git","GitHub","Gauss","Amazon Cloud Computing Services","Hadoop","Databases","Latex")
langies <- c("C","C++","CUDA","OpenCL","Java","Javascript","SQL","Fortran","Julia","Scala")
respies <-  
  c("The course sounded really interesting/useful", 
    "I have to take the course to satisfy a requirement for my degree",
    "My advisor told me to take the course", 
    "I needed more units and it seemed like the best available course",
    "I am hoping that the course content will be helpful for my research",
    "I am hoping that the course content will be helpful for my future job prospects")

table(pcs[,2]) # 14 Stat, 6 Other
table(pcs[,3]) # 3 MS, 17 PhD

table(pcs[,4]) # 
count_responses(pcs[,4],cats=courses)

table(pcs[,5])
count_responses(pcs[,5],cats=topics)

table(pcs[,6]) # Only Windows = 9
count_responses(pcs[,6],cats=c("Linux","Windows","Mac OS X"))

table(pcs[,7])
table(pcs[,8])

count_responses(pcs[,9],cats=compies)
count_responses(pcs[,10],cats=langies,fixed=TRUE)

table(pcs[,11])

count_responses(pcs[,12],cats=respies)

qplot(Coffee.or.tea.,data=pcs) # beh. ugly

table(pcs$Coffee.or.tea.)
table(pcs$Cats.or.dogs.)

mosaicplot(xtabs(~Cats.or.dogs.+Coffee.or.tea.,data=pcs))

