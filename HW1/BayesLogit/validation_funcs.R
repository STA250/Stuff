
##
## Validation functions
##

"cover_func" <- function(truth,X,Y=NULL,type="one-sided",return.type=c('all','proportions','indicators'),fixed.dims=NULL,verbose=FALSE)
{
  ####
  ##
  ## truth :: list of (vector of) parameters, each element corresponding to
  ##          to a different parameter. 
  ##          **NOTE** must be a vector not a matrix.
  ##
  ## X     :: list of posterior quantiles for the same (vector of) parameters,
  ##          each element of the list corresponds to a different validation run.
  ##          if type=='two-sided', X refers to the upper end of the interval.
  ##          Failed datasets can be handled by including them as NULL 
  ##          elements in the list.
  ##          **NOTE** must have nrow == length(truth)
  ##
  ## Y     :: required only if type=='two-sided'. list of quantiles for the 
  ##          same (vector of) parameters. Note that X refers to the upper 
  ##         end of the interval, Y to the lower end.
  ##
  ## type  :: either 'one-sided', in which case it computes:
  ##            \sum Pr(truth <= X)
  ##          or, 'two-sided', in which case it computes:
  ##            \sum Pr(Y <= truth <= X)
  ##
  ## fixed.dims :: if the length of the parameter varies from iteration
  ##          to iteration, then fixed.dim.prop should specify the indices of
  ##          the parameters that do not vary. All other parameters will be
  ##          averaged over.
  ##
  ## Notes :: this function is designed to be called separately for each
  ##          desired quantile, and provides options for both interval-based
  ##          and quantile-based coverage.
  ##
  ## Return value ::
  ##    Three options:
  ##    (all) returns (i) a list of indicator variables corresponding to 'type',
  ##          the proportion of 1's, and the vector of successful indices.
  ##   
  ##    
  ## 
  ####
  
  foo <- list(NULL)
  successful_dataset_indices <- NULL
  
  if (verbose)
    cat("Checking dimensions of inputs...\n")

  # Check X:
  for (j in 1:length(X)){
    
    if (!is.null(X[[j]]) && !is.null(truth[[j]])){
      if (length(truth[[j]]) != nrow(X[[j]])){
        if (verbose){
          cat(paste("length(truth[[",j,"]]) = (",paste(length(truth[[j]]),collapse=","),")\n",sep=""))
          cat(paste("dim(X[[",j,"]])     = (",paste(dim(X[[j]]),collapse=","),")\n",sep=""))
        }
        stop("dimension mismatch between 'truth' and 'qlist'")
      }
    }
  }
  # Check Y:
  if (type=='two-sided'){
    
    if (is.null(Y)) {
      stop("Y object (lower boundary) is required when type=='two-sided'.")
    }
    
    if (length(X) != length(Y)) {
      stop("Length of X object must equal length of Y object.")
    }
    
    for (j in 1:length(Y)){
            
      if (!is.null(Y[[j]]) && !is.null(truth[[j]])){
        if (length(truth[[j]]) != nrow(Y[[j]])){
          if (verbose){
            cat(paste("length(truth[[",j,"]]) = (",paste(length(truth[[j]]),collapse=","),")\n",sep=""))
            cat(paste("dim(Y[[",j,"]])     = (",paste(dim(Y[[j]]),collapse=","),")\n",sep=""))
          }
          stop("dimension mismatch between 'truth' and Y 'qlist'")
        }
      }
    } 
  } #END check Y.

  if (verbose)
    cat("dimension check passed. Computing indicators...\n")

  if (type=='one-sided'){
    
    for (j in 1:length(X)){
      if (is.null(X[[j]]) || is.null(truth[[j]])){
        foo[[j]] <- NULL
      } else {
        successful_dataset_indices <- c(successful_dataset_indices,j)
        foo[[j]] <- (truth[[j]]<=X[[j]])
      }
    }
  
  } else {
    if (type=='two-sided'){

      for (j in 1:length(X)){
        if (is.null(X[[j]]) || is.null(truth[[j]])){
          foo[[j]] <- NULL
        } else {
          successful_dataset_indices <- c(successful_dataset_indices,j)
          foo[[j]] <- (Y[[j]]<=truth[[j]])&(truth[[j]]<=X[[j]])
        }
      }
      
    } else {
      stop("unknown 'type' argument specified. must be 'one-sided' or 'two-sided'")
    }
  }
  
  if (verbose)
    cat("Indicators successfully computed...\n")

  if (return.type=='indicators')
    return(foo)
  
  # Must either return summed version, or both:
  
  if (verbose)
    cat("Computing coverage proportions...\n")

  if (!is.null(fixed.dims)){

    if (all(fixed.dims != 0)){

      num_cover <- 0
      for (i in successful_dataset_indices){
        num_cover <- num_cover + foo[[i]][fixed.dims,,drop=FALSE]
      }
      prop_cover <- num_cover/length(successful_dataset_indices)

    } else {

      prop_cover <- NULL

    }

    num_cover_varying <- 0
    total_num_par <- 0
    for (i in successful_dataset_indices){
      tmp.cover <- matrix(apply(foo[[i]][-fixed.dims,,drop=FALSE],2,sum),nrow=1)
      num_cover_varying <- num_cover_varying + tmp.cover
      total_num_par <- total_num_par + nrow(foo[[i]][-fixed.dims,,drop=FALSE])
    }
    prop_cover_varying <- num_cover_varying/total_num_par

    prop_cover <- rbind(prop_cover,prop_cover_varying)

  } else {

    num_cover <- 0
    for (i in successful_dataset_indices){
      num_cover <- num_cover + foo[[i]]
    }
    prop_cover <- num_cover/length(successful_dataset_indices)

  }

  if (verbose)
    cat("Coverage proportions successfully computed...\n")

  if (return.type=='proportions')
    return(prop_cover)

  if (return.type!='all')
    stop("Invalid return type, must be one of 'all', 'proportions' or 'indicators'")
  
  if (return.type=="all")
    return(list("indicators"=foo,"successful_dataset_indices"=successful_dataset_indices,"proportions"=prop_cover))
}

"posterior_interval_plot" <- function(truth,successful_dataset_indices,
    point.est=NULL,interval.lo=NULL,interval.hi=NULL,
    xlab="True parameter",ylab="Posterior",main="Posterior intervals vs. Truth",
    pch=16,pcol="black",cex.pts=1.0,lwd.seg=0.15,lcol="gray",flagcol="blue")
{
  ####
  ## truth       :: vector, true values of the parameter
  ## point.est   :: vector, usually posterior median, or posterior mean
  ## interval.lo :: vector, lower end of the interval
  ## interval.hi :: vector, upper end of the interval
  ## successful_dataset_indices :: vector, either indices or T/F
  ####

  xlo <- min(cbind(point.est,truth,interval.lo)[successful_dataset_indices,])-0.1
  xhi <- max(cbind(point.est,truth,interval.hi)[successful_dataset_indices,])+0.1
  ylo <- xlo
  yhi <- xhi
  
  main <- paste(main," (",length(successful_dataset_indices)," datasets)",sep="")
  
  plot(NULL,xlim=c(xlo,xhi),ylim=c(ylo,yhi),ylab=ylab,xlab=xlab,main=main)

  if (!is.null(interval.lo) && !is.null(interval.hi)){
        
    # Mark bad intervals with another color
    newcol = rep(lcol,length(successful_dataset_indices))
    ix.bad <- (interval.lo > truth) | (interval.hi < truth)
    newcol[ix.bad] = flagcol
    
    # Plot all intervals
    segments(x0=truth[successful_dataset_indices],
         y0=interval.hi[successful_dataset_indices],
         x1=truth[successful_dataset_indices],
         y1=interval.lo[successful_dataset_indices],
         lwd=lwd.seg,col=newcol)
    
    # Mark bad intervals for visibility
    legend("bottomright", legend=paste(sum(ix.bad,na.rm=TRUE),"flagged\nintervals"))
  }

  if (!is.null(point.est)){
    points(y=point.est[successful_dataset_indices],
       x=truth[successful_dataset_indices],
       col=pcol,pch=pch,cex=cex.pts)
  }

  abline(a=0,b=1,lwd=1.2,col="red")

  return()
}


"generic_posterior_interval_plot" <- function(qlist,truth,nameid,names,successful_dataset_indices,
    point.est=NULL,conf.level=0.95) 
{
  ####
  ## qlist       :: list of matrices of the parameters 
  ## truth       :: vector, true values of the parameter
  ## nameid      :: vector, index of partial parameter vector
  ## names       :: vector, names of desired parameters
  ## point.est   :: vector, usually posterior median, or posterior mean
  ## successful_dataset_indices :: vector, either indices or T/F
  ####
  
  ci.lo.int <- (1-conf.level)*100
  ci.hi.int <- (conf.level)*100
  
  m <- length(nameid)
  
  for(j in 1:m){
    cat("Creating vector of true values of parameter...\n")
    true.val <- unlist(lapply(truth,function(x){if (!is.null(x) && length(x)>=2){x[nameid[j]]} else {NA}}))
    
    cat("Computing central (1-conf.level)*100% posterior intervals...\n")
    q.lo <- unlist(lapply(qlist,function(x){if (!is.null(x) && ncol(x)>=ci.lo.int){x[names[j],ppaste(ci.lo.int,"%")]} else {NA}}))
    q.hi <- unlist(lapply(qlist,function(x){if (!is.null(x) && ncol(x)>=ci.lo.int){x[names[j],ppaste(ci.hi.int,"%")]} else {NA}}))
    
    cat("Making posterior intervals vs truth plot...\n")
    posterior_interval_plot(true.val,successful_dataset_indices,
                            point.est=NULL,interval.lo=q.lo,interval.hi=q.hi,lwd.seg=0.40,
                            xlab=c("True ",names[j]),ylab="Posterior",main=ppaste(ci.hi.int,"% Posterior intervals vs. Truth")  )
  }
}


"global_coverage_plot" <- function(desired_coverage,actual_coverage,bands=FALSE,num_sims=NA,band.width=0.95,col=NULL,lwd=1.2,xlab="Nominal",ylab="Actual",main="default")
{

  if (!is.matrix(actual_coverage) || ncol(actual_coverage)<1)
    stop("'actual_coverage' must be a matrix with at least column")
  if (nrow(actual_coverage) != length(desired_coverage))
    stop("number of rows of 'actual_coverage' must match the length of 'desired_coverage'")

  # Sort out the colors of the lines:
  npars <- ncol(actual_coverage)
  if (length(col)==0){
    col <- rainbow(npars)
  }
  if (length(col)<1){
    input.col <- col
    col <- rep(NA,npars)
    col[] <- input.col
  }
  
  # Sort the ordering in case not done by user:
  dss <- sort(desired_coverage,index.return=TRUE)
  desired_coverage <- dss$x
  actual_coverage <- actual_coverage[dss$ix,]

	if (main=="default"){
		main <- "Actual vs. Nominal Coverage"
		if (!is.na(num_sims)){
			main <- paste(main," (",num_sims," datasets)",sep="")
		}
	}

  plot(y=100*actual_coverage[,1],x=100*desired_coverage,type="n",
     xlim=c(0,100),ylim=c(0,100),ylab=ylab,xlab=xlab,main=main)

  for (i in 1:npars){
    lines(y=100*actual_coverage[,i],x=100*desired_coverage,type="l",col=col[i],lwd=lwd)
  }

	if (bands){
    if (is.na(num_sims)){
			stop("'num_sims' must be specified if bands=TRUE")
		}
    ll <- (1.0-band.width)/2
    ul <- band.width + ll
		lr <- qbinom(p=ll,size=num_sims,prob=desired_coverage)/num_sims
		ur <- qbinom(p=ul,size=num_sims,prob=desired_coverage)/num_sims
		lines(x=100*desired_coverage,y=100*ur,col="black",lty=2)
		lines(x=100*desired_coverage,y=100*lr,col="black",lty=2)
	}

  abline(a=0,b=1,col="black",lwd=1.5)

  legend(x=10,y=90,col=c("black",col),legend=c("Target",colnames(actual_coverage)),lty=1)

  return()
}


