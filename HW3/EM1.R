"bisection" <- function(func,l,u,epsilon,iteration,verbose=T){
  converged <- F
  count=0
  while(!converged){
    c<-(l+u)/2
    if (abs(func(c))<epsilon) converged<-T     else {
      if (func(l)*func(c)<0) u<-c
      else l<-c
    }
     count<-count+1
   
    if (!(count<iteration)){
      if (verbose) cat("exceed iteration limit!\n")
      break
    }
    if (verbose) cat(paste("l is", l,", u is",u,", and func(c) is", func(c),"\n" ))
  }
  c
}  

"NewtonRaphson" <- function(func,deriv.func,init,epsilon,iteration,verbose){
  converged <- F
  count=0
  x <- init
  while(!converged){
  if (abs(func(x))<epsilon) converged<-T     else   x <- x-func(x)/deriv.func(x)
  count<-count+1
  if (!(count<iteration)) {
    if (verbose) cat("exceed iteration limit!\n")
    break
  }
  if (verbose) cat(paste("func(x) is", func(x),"\n" ))
  }
  
  x
}


"func" <-function(lamda){
  value<-68+15*lamda-197*lamda^2  
  value
}

deriv.func<-function(lamda){
 df<-15-394*lamda
  df
}


b_root<-bisection(func=func,l=0,u=1,epsilon=10^(-10),iteration=100,verbose=T)
nr_root<-NewtonRaphson(func,deriv.func,init=0.6,epsilon=10^(-10),iteration=100,verbose=T)

cat(paste("root of bisection method is",b_root,"\n"))
cat(paste("root of NR method is",nr_root,"\n"))

