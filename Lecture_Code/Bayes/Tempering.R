
rm(list=ls())

# MH and Parallel Tempering Examples

MH <- function( logTargetDensityFunc , logProposalDensityFunc , proposalNewFunc , nIters , startingValue , isSymmetric = FALSE , storeEveryNthDraw = 1 , nChains = 1 ){
  # Set everything up...
  binaryAcceptance <- matrix( 0 , ncol = nIters , nrow = nChains ) ; binaryAcceptance[,1] <- 1
  # Since we always accept the starting state.
  # The binary acceptance data is stored in a matrix with each row corresponding to a single chain, each column to an iteration.
  storedDraws <- array( startingValue , dim = c( length(startingValue)/nChains , nIters , nChains ) )
  dimension <- length(startingValue)/nChains
  # Starting value is a vector of the starting values for all nChains, which can be different.
  # Stores the starting values, and initializes the storage matrix, the starting value will usually be lost after burn-in.
  # First do Metropolis algorithm...
  if ( isSymmetric == TRUE ){
    for ( k in 1:nChains ){
      currentState <- startingValue[ (dimension*(k-1) + 1 ):( dimension*k )]
      for ( i in 1:( (nIters*storeEveryNthDraw) - 1) ){# Propose a new value.
        proposedState <- proposalNewFunc( currentState )
        # The acceptance step:
        if ( runif(1) < exp( logTargetDensityFunc( proposedState ) - logTargetDensityFunc ( currentState ) ) ){ currentState <- proposedState }
        if ( i %% storeEveryNthDraw == 0 ) { # For a multiple of n, store the draw...
          # If the current state is not the same the same as the last saved state then we have accepted a move:
          if ( any( !( storedDraws[ , (i/storeEveryNthDraw) , k ] == currentState ) ) ){
            binaryAcceptance[ k , ( i / storeEveryNthDraw ) ] <- 1 }
          storedDraws[ , (i / storeEveryNthDraw) + 1 , k ] <- currentState
        }
      }
    }
  } else { # Finishes the Metropolis algorithm, now do the Metropolis-Hastings algorithm...
    for (k in 1:nChains){
      currentState <- startingValue[ (dimension*(k-1) + 1):( dimension*k )]
      for (i in 1:( (nIters*storeEveryNthDraw) - 1 ) ){# Propose a new value.
        proposedState <- proposalNewFunc( currentState )
        # The acceptance step:
        if ( runif(1) < exp( logTargetDensityFunc( proposedState ) + logProposalDensityFunc( proposedState , currentState ) -
                             logTargetDensityFunc( currentState  ) - logProposalDensityFunc( currentState , proposedState ) ) ){
          if ( i %% storeEveryNthDraw == 0 ){
            if ( any( !( storedDraws[ , (i/storeEveryNthDraw) , k ] == currentState ) ) ){
              binaryAcceptance[ k , ( i / storeEveryNthDraw ) ] <- 1
            }
            storedDraws[ , (i/storeEveryNthDraw) + 1 , k ] <- currentState
          }
        }
      }
    }
  }
  return( list( binaryAcceptance , storedDraws ) )
}

parallelTempering <- function( temperatureLadder , logTargetDensityFunc , logMutationProposalDensityFunc,
                               mutationProposalNewFunc , isSymmetricVec = NULL, movesMixtureVec,
                               nIters , startingValues , storeEveryNthDraw = 1 )
{
  binaryARM <- matrix( 2 , nrow = length( temperatureLadder ) , ncol = nIters*storeEveryNthDraw )
  binaryARE <- matrix( 2 , nrow = length( temperatureLadder ) - 1 , ncol = nIters*storeEveryNthDraw )
  # Each row of binaryAcceptance corresponds to a temperature.
  dimension <- length( startingValues )/length( temperatureLadder )
  storedDraws <- matrix( 0 , nrow = dimension , ncol = nIters )
  storedDraws[ , 1 ] <- startingValues[ ( length( startingValues ) - dimension + 1 ):( length( startingValues ) ) ]
  # We will store the starting state as first draw.
  currentStates <- matrix( startingValues , ncol = length( temperatureLadder ) )
  # Create a function to generate all proposed states in one go:
  allLevelProposer <- function( xval , temperatureLadder ){
    bigProposalVector <- NULL
    for ( j in 1:length( temperatureLadder ) ){
      bigProposalVector <- cbind( bigProposalVector , mutationProposalNewFunc( xval[ , j ] , temperatureLadder[ j ] ) )
    }
    bigProposalVector
  }
  # Now we are ready to start the algorithm...
  for ( i in 1:( nIters*storeEveryNthDraw - 1 ) ){
    # Do one iteration...
    # First decide whether to do mutation:
    if ( runif( 1 ) < movesMixtureVec[ 1 ] ){ # Do mutation:
      # Mutations at different levels are independent so we will do them all in one go...
      proposedMutationStates <- allLevelProposer( currentStates , temperatureLadder ) ### ; print("Proposed states are:") ; print(proposedMutationStates)
      computationStep <- matrix( apply( cbind( proposedMutationStates , currentStates ) , 2 , logTargetDensityFunc ) , ncol = 2*length( temperatureLadder ) )
###      print("Computation step is:") ; print(computationStep)
      updateColumns <- NULL
      comparisonBlock <- currentStates
      for ( j in 1:length( temperatureLadder ) ){                       ###  ; print(paste("Mutation at level:",j))
        if ( log( runif( 1 ) ) < ( computationStep[ , j ] - computationStep[ , j + length( temperatureLadder ) ] +
                                   logMutationProposalDensityFunc( proposedMutationStates[ , j ] , currentStates[ , j ] , temperatureLadder[ j ] ) -
                                   logMutationProposalDensityFunc( currentStates[ , j ] , proposedMutationStates[ , j ] , temperatureLadder[ j ] ) ) ){
          updateColumns <- c( updateColumns , j )
        } }
      # Update all the states that were accepted:
      currentStates[ , updateColumns ] <- proposedMutationStates[ , updateColumns ]
      # Note that this stores the acceptance/reject histories stores for EVERY iteration, even if we only STORE EVERY Nth DRAW.
      # This is because the definitions and the histories needed for the problem set don't always make sense when saving only every nth draw.
      binaryARM[ , i ] <- ifelse ( apply( !( currentStates == comparisonBlock ) , 2 , any ) , 1 , 0 )
      if ( ( i %% storeEveryNthDraw ) == 0 ){
      # Store the state of the desired temperature chain:
      storedDraws[ , ( i / storeEveryNthDraw ) ] <- currentStates[ , length( temperatureLadder ) ]
    }
  } else { # Finished the mutation step, now for the exchange...
      comparisonBlock <- currentStates
      ijExchangePropose <- matrix( 0 , nrow = length( temperatureLadder ) , ncol = 2 )
      possibleCombinations <- cbind( 1:( length( temperatureLadder ) - 1 ) , 2:( length( temperatureLadder ) ) )
      for ( w in 1:( length( temperatureLadder ) - 1 ) ){         ### ; print(paste("Exchange number:",w))
        # Apply systematically to every combination of chains, starting at the hottest (recall that temperatures are supplied in descending order)...
        if ( log( runif ( 1 ) ) <
            temperatureLadder[ length( temperatureLadder ) ]*( logTargetDensityFunc( currentStates[ , possibleCombinations[ w , 1 ] ] ) -
                                                               logTargetDensityFunc( currentStates[ , possibleCombinations[ w , 2 ] ] ) )*
            ( ( 1 / temperatureLadder[ possibleCombinations[ w , 2 ] ] ) - ( 1 / temperatureLadder[ possibleCombinations[ w , 1 ] ] ) ) ){
        currentStates[ , possibleCombinations[ w , 1:2 ] ] <- currentStates[ , possibleCombinations[ w , 2:1 ] ] # Switch the states across chains.
      }
      } # Repeats for each possible switch.
      # Note that this stores the acceptance/reject histories stores for EVERY iteration, even if we only STORE EVERY Nth DRAW.
      # This is because the definitions and the histories needed for the problem set don't always make sense when saving only every nth draw.
      binaryARE[ unique( possibleCombinations[ , 1 ] ) , i ] <-
        ifelse ( apply( matrix( !( currentStates[ , unique( possibleCombinations[ , 1 ] ) ] == comparisonBlock[ , unique( possibleCombinations[ , 1 ] ) ] ) , ncol = length( unique( possibleCombinations[ , 1 ] ) ) ) , 2 , any ) , 1 , 0 )
      if ( ( i %% storeEveryNthDraw ) == 0 ){
###        print("ijExchangePropose =") ; print(ijExchangePropose)
###        print("Current states are:") ; print(currentStates)
###        print("Comparison block is:") ; print(comparisonBlock)
      storedDraws[ , ( i / storeEveryNthDraw ) ] <- currentStates[ , length( temperatureLadder ) ]
    } # Records the Random Exchange accept/reject/not proposed events at iteration i.
    } # Completes the Random Exchange step.
    } # Repeats the desired number of times.
  return( list( binaryARM , binaryARE , storedDraws ) )
}

# ------------------------------------------------
# --- Example : Mixture Tempering -------------
# ------------------------------------------------

# --- (a) Using ordinary MH ----------------------

mixtureNormLogTarget1 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 1 , sd = 1 , log = FALSE ) ) ) }
mixtureNormLogTarget2 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 2 , sd = 1 , log = FALSE ) ) ) }
mixtureNormLogTarget3 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 3 , sd = 1 , log = FALSE ) ) ) }
mixtureNormLogTarget4 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 4 , sd = 1 , log = FALSE ) ) ) }
mixtureNormLogTarget5 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 5 , sd = 1 , log = FALSE ) ) ) }
mixtureNormLogTarget6 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 6 , sd = 1 , log = FALSE ) ) ) }
mixtureNormLogTarget7 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 7 , sd = 1 , log = FALSE ) ) ) }
mixtureNormLogTarget8 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 8 , sd = 1 , log = FALSE ) ) ) }
mixtureNormLogTarget9 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 9 , sd = 1 , log = FALSE ) ) ) }
mixtureNormLogTarget10 <- function( x ){ log( 0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 10 , sd = 1 , log = FALSE ) ) ) }
logNormPropDensity <- function( x , y ){ dnorm( y , mean = x , sd = sqrt( 3 ) , log = TRUE ) }
normPropFunction <- function( x ){ rnorm( 1 , mean = x , sd = sqrt( 3 ) ) }
startState <- 0
cat("Time of MH run:\n")
print(system.time(
            Runs1 <- MH( mixtureNormLogTarget1 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
            ))
# Takes 0.17 of a second.
cat("Now running 9 more of them...\n")
Runs2 <- MH( mixtureNormLogTarget2 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
Runs3 <- MH( mixtureNormLogTarget3 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
Runs4 <- MH( mixtureNormLogTarget4 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
Runs5 <- MH( mixtureNormLogTarget5 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
Runs6 <- MH( mixtureNormLogTarget6 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
Runs7 <- MH( mixtureNormLogTarget7 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
Runs8 <- MH( mixtureNormLogTarget8 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
Runs9 <- MH( mixtureNormLogTarget9 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
Runs10 <- MH( mixtureNormLogTarget10 , logNormPropDensity ,normPropFunction , 2000 , startState , isSymmetric = TRUE , nChains = 1 , storeEveryNthDraw = 1 )
cat("done.\n")

pdf("MixtureMH.pdf")
par(mfrow=c(2,2))
# plot(Runs1[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=1")
# acf(Runs1[[2]][1,,1],main="Autocorrelation: Mixture mu=1") # Works ok.
# plot(Runs2[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=2")
# acf(Runs2[[2]][1,,1],main="Autocorrelation: Mixture mu=2") # Still ok, worsening aotocorrelation though.
# plot(Runs3[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=3")
# acf(Runs3[[2]][1,,1],main="Autocorrelation: Mixture mu=3") # Still just ok.
# plot(Runs4[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=4")
# acf(Runs4[[2]][1,,1],main="Autocorrelation: Mixture mu=4") # Still just ok.
# plot(Runs5[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=5")
# acf(Runs5[[2]][1,,1],main="Autocorrelation: Mixture mu=5") # Getting pretty bad.
# plot(Runs6[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=6")
# acf(Runs6[[2]][1,,1],main="Autocorrelation: Mixture mu=6") # Absolutely rubbish now.
plot(Runs7[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=7")
acf(Runs7[[2]][1,,1],main="Autocorrelation: Mixture mu=7") # Even worse.
# plot(Runs8[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=8")
# acf(Runs8[[2]][1,,1],main="Autocorrelation: Mixture mu=8") # Only just mixing.
# plot(Runs9[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=9") # Never reaches second mode.
# acf(Runs9[[2]][1,,1],main="Autocorrelation: Mixture mu=9")
plot(Runs10[[2]][1,,1],type="l",xlab="Iteration",ylab="x",main="Trace plot: Mixture mu=10") # Never reaches second mode.
acf(Runs10[[2]][1,,1],main="Autocorrelation: Mixture mu=10")
dev.off()

# --- (b) Using PT -------------------------

# The log of the two target denisties for mu_1 and mu_2 chosen to be ?? and ?? respectively.
mu1 <- 7 ; mu2 <- 10

# The proposal density functions we need.
logMuteNormPropDenFunc <- function( x , y , temperature ){ dnorm( y , mean = x , sd = sqrt( 3*sqrt( temperature ) ) , log = TRUE ) }
mutatePropNormFunction <- function( x , temperature ){ rnorm( 1 , mean = x , sd = 3*sqrt( temperature ) ) } 

# The mixture probability vector.
mixtureNormMixtureProb <- c( 0.3 , 0.7 )

# The temperature ladder...
xPointsForGraphs <- matrix(seq(-50,50,0.01))

mixtureNormTargetMean7Power0 <- function( x ){(0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 7 , sd = 1 , log = FALSE ) )) }
mixtureNormTargetMean7Power1 <- function( x ){(0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 7 , sd = 1 , log = FALSE ) ))^(1/10) }
mixtureNormTargetMean7Power2 <- function( x ){(0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 7 , sd = 1 , log = FALSE ) ))^(1/20) }
mixtureNormTargetMean7Power3 <- function( x ){(0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 7 , sd = 1 , log = FALSE ) ))^(1/40) }

png("PowersMean7.png")
par(mfrow=c(2,2))
  unnormalizedPointsMean7Pow0 <- apply(xPointsForGraphs,1,mixtureNormTargetMean7Power0)
  normalizingConstantMean7Pow0 <- 0.01*sum(unnormalizedPointsMean7Pow0[-10001])
  normalizedPoints7Pow0 <- unnormalizedPointsMean7Pow0/normalizingConstantMean7Pow0
  plot(y=normalizedPoints7Pow0,x=xPointsForGraphs,type="l",xlab="x",ylab="f(x)",main="Target (mu=7): t=1")
  unnormalizedPointsMean7Pow1 <- apply(xPointsForGraphs,1,mixtureNormTargetMean7Power1)
  normalizingConstantMean7Pow1 <- 0.01*sum(unnormalizedPointsMean7Pow1[-10001])
  normalizedPoints7Pow1 <- unnormalizedPointsMean7Pow1/normalizingConstantMean7Pow1
  plot(normalizedPoints7Pow1,x=xPointsForGraphs,type="l",xlab="x",ylab="f(x)",main="Target (mu=7): t=10")
  unnormalizedPointsMean7Pow2 <- apply(xPointsForGraphs,1,mixtureNormTargetMean7Power2)
  normalizingConstantMean7Pow2 <- 0.01*sum(unnormalizedPointsMean7Pow2[-10001])
  normalizedPoints7Pow2 <- unnormalizedPointsMean7Pow2/normalizingConstantMean7Pow2
  plot(normalizedPoints7Pow2,x=xPointsForGraphs,type="l",xlab="x",ylab="f(x)",main="Target (mu=7): t=20")
  unnormalizedPointsMean7Pow3 <- apply(xPointsForGraphs,1,mixtureNormTargetMean7Power3)
  normalizingConstantMean7Pow3 <- 0.01*sum(unnormalizedPointsMean7Pow3[-10001])
  normalizedPoints7Pow3 <- unnormalizedPointsMean7Pow3/normalizingConstantMean7Pow3
  plot(normalizedPoints7Pow3,x=xPointsForGraphs,type="l",xlab="x",ylab="f(x)",main="Target (mu=7): t=40")
dev.off()

mixtureNormTargetMean10Power0 <- function( x ){(0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 10 , sd = 1 , log = FALSE ) )) }
mixtureNormTargetMean10Power1 <- function( x ){(0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 10 , sd = 1 , log = FALSE ) ))^(1/10) }
mixtureNormTargetMean10Power2 <- function( x ){(0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 10 , sd = 1 , log = FALSE ) ))^(1/40) }
mixtureNormTargetMean10Power3 <- function( x ){(0.5*( dnorm( x , sd = 1 , log = FALSE ) + dnorm( x , mean = 10 , sd = 1 , log = FALSE ) ))^(1/60) }

png("PowersMean10.png")
par(mfrow=c(2,2))
  unnormalizedPointsMean10Pow0 <- apply(xPointsForGraphs,1,mixtureNormTargetMean10Power0)
  normalizingConstantMean10Pow0 <- 0.01*sum(unnormalizedPointsMean10Pow0[-10001])
  normalizedPoints10Pow0 <- unnormalizedPointsMean10Pow0/normalizingConstantMean10Pow0
  plot(normalizedPoints10Pow0,x=xPointsForGraphs,type="l",xlab="x",ylab="f(x)",main="Target (mu=10): t=1")
  unnormalizedPointsMean10Pow1 <- apply(xPointsForGraphs,1,mixtureNormTargetMean10Power1)
  normalizingConstantMean10Pow1 <- 0.01*sum(unnormalizedPointsMean10Pow1[-10001])
  normalizedPoints10Pow1 <- unnormalizedPointsMean10Pow1/normalizingConstantMean10Pow1
  plot(normalizedPoints10Pow1,x=xPointsForGraphs,type="l",xlab="x",ylab="f(x)",main="Target (mu=10): t=10")
  unnormalizedPointsMean10Pow2 <- apply(xPointsForGraphs,1,mixtureNormTargetMean10Power2)
  normalizingConstantMean10Pow2 <- 0.01*sum(unnormalizedPointsMean10Pow2[-10001])
  normalizedPoints10Pow2 <- unnormalizedPointsMean10Pow2/normalizingConstantMean10Pow2
  plot(normalizedPoints10Pow2,x=xPointsForGraphs,type="l",xlab="x",ylab="f(x)",main="Target (mu=10): t=40")
  unnormalizedPointsMean10Pow3 <- apply(xPointsForGraphs,1,mixtureNormTargetMean10Power3)
  normalizingConstantMean10Pow3 <- 0.01*sum(unnormalizedPointsMean10Pow3[-10001])
  normalizedPoints10Pow3 <- unnormalizedPointsMean10Pow3/normalizingConstantMean10Pow3
  plot(normalizedPoints10Pow3,x=xPointsForGraphs,type="l",xlab="x",ylab="f(x)",main="Target (mu=10): t=60")
dev.off()

# So I will choose 40 as my maximum temperature because it enables sufficient movement about the density.
# Try the linear-in-log form of the temperature ladder.

# The PT function is fairly quick so I'll use eight temperatures.
# So decide on the following temperature ladders by using the above:

lu <- log(40) ; lu2 <- log(60)
mixNormTempLadder7 <- c( 40 ,6*lu/7 , 5*lu/7 , 4*lu/7 , 3*lu/7 , 2*lu/7 , lu/7 , 1 )
mixNormTempLadder10 <- c( 60 ,6*lu/7 , 5*lu/7 , 4*lu/7 , 3*lu/7 , 2*lu/7 , lu/7 , 1 )

# The starting values.
mixNormStartingValues7 <- rep( 0 , times = length( mixNormTempLadder7 ) )
mixNormStartingValues10 <- rep( 0 , times = length( mixNormTempLadder10 ) )

system.time(
            mixNormPT1 <- parallelTempering( mixNormTempLadder7 , mixtureNormLogTarget7 , logMuteNormPropDenFunc , mutatePropNormFunction , isSymmetricVec = NULL , mixtureNormMixtureProb , 2000 , mixNormStartingValues7 , storeEveryNthDraw = 1 )
            )
system.time(
            mixNormPT2 <- parallelTempering( mixNormTempLadder10 , mixtureNormLogTarget10 , logMuteNormPropDenFunc , mutatePropNormFunction , isSymmetricVec = NULL , mixtureNormMixtureProb , 2000 , mixNormStartingValues10 , storeEveryNthDraw = 1 )
            )

# Takes 2.2 seconds for each chain...
dim(mixNormPT1[[1]]) ; dim(mixNormPT1[[2]]) ; dim(mixNormPT1[[3]])
png("ptPreBurnIn.png")
par(mfrow=c(2,2))
plot(as.vector(mixNormPT1[[3]]),type="l",xlab="Iteration",ylab="x",main="Trace plot (PT): Mixture mu=7") 
acf(as.vector(mixNormPT1[[3]]),main="Autocorrelation (PT): Mixture mu=7")
plot(as.vector(mixNormPT2[[3]]),type="l",xlab="Iteration",ylab="x",main="Trace plot (PT): Mixture mu=10")
acf(as.vector(mixNormPT2[[3]]),main="Autocorrelation (PT): Mixture mu=10")
dev.off()

# 500 burn-in for both looks okay. 
png("ptPostBurnIn.png")
par(mfrow=c(2,2))
plot(as.vector(mixNormPT1[[3]][,501:2000]),type="l",xlab="Iteration",ylab="x",main="Trace plot (PT): Mixture mu=7")
acf(as.vector(mixNormPT1[[3]][,501:2000]),main="Autocorrelation (PT): Mixture mu=7")
plot(as.vector(mixNormPT2[[3]][,501:2000]),type="l",xlab="Iteration",ylab="x",main="Trace plot (PT): Mixture mu=10")
acf(as.vector(mixNormPT2[[3]][,501:2000]),main="Autocorrelation (PT): Mixture mu=10")
dev.off()

dim(mixNormPT1[[3]])

library(MASS)

png("TwoThouItsPT.png")
par(mfrow=c(1,2))
truehist(as.vector(mixNormPT1[[3]][,501:2000]),nbins=30,xlab="x",ylab="Frequency",main="Samples using PT: mu=7")
truehist(as.vector(mixNormPT2[[3]][,501:2000]),nbins=30,xlab="x",ylab="Frequency",main="Samples using PT: mu=10")
dev.off()

#----------------------------------------------
# Try for 10,000 draws, saving every 10th draw.
#----------------------------------------------

cat("Running PT for 10,000 iterations...\n")
print(system.time({
  mixNormPT1LONG <- parallelTempering( mixNormTempLadder7 , mixtureNormLogTarget7 , logMuteNormPropDenFunc , mutatePropNormFunction , isSymmetricVec = NULL , mixtureNormMixtureProb , 10000 , mixNormStartingValues7 , storeEveryNthDraw = 10 )
}))

cat("Again running PT for 10,000 iterations...\n")
print(system.time({
  mixNormPT2LONG <- parallelTempering( mixNormTempLadder10 , mixtureNormLogTarget10 , logMuteNormPropDenFunc , mutatePropNormFunction , isSymmetricVec = NULL , mixtureNormMixtureProb , 10000 , mixNormStartingValues10 , storeEveryNthDraw = 10 )
}))

# Takes 90 seconds for each chain...
png("ptPreBurnInlong.png")
par(mfrow=c(2,2))
plot(as.vector(mixNormPT1LONG[[3]]),type="l",xlab="Iteration",ylab="x",main="Trace plot (PT): Mixture mu=7") 
acf(as.vector(mixNormPT1LONG[[3]]),main="Autocorrelation (PT): Mixture mu=7")
plot(as.vector(mixNormPT2LONG[[3]]),type="l",xlab="Iteration",ylab="x",main="Trace plot (PT): Mixture mu=10")
acf(as.vector(mixNormPT2LONG[[3]]),main="Autocorrelation (PT): Mixture mu=10")
dev.off() # Much better!

# 1000 burn-in for both looks okay. Asked to plot post-burn-in graphs...
pdf("MixturePT.pdf")
par(mfrow=c(2,2))
plot(as.vector(mixNormPT1LONG[[3]][,1001:10000]),type="l",xlab="Iteration",ylab="x",main="Trace plot (PT): Mixture mu=7") 
acf(as.vector(mixNormPT1LONG[[3]][,1001:10000]),main="Autocorrelation (PT): Mixture mu=7")
plot(as.vector(mixNormPT2LONG[[3]][,1001:10000]),type="l",xlab="Iteration",ylab="x",main="Trace plot (PT): Mixture mu=10") 
acf(as.vector(mixNormPT2LONG[[3]][,1001:10000]),main="Autocorrelation (PT): Mixture mu=10")
dev.off()

dim(mixNormPT1LONG[[3]])

library(MASS)

png("TwoThouItsPTlong.png")
par(mfrow=c(1,2))
truehist(as.vector(mixNormPT1LONG[[3]][,1001:10000]),nbins=40,xlab="x",ylab="Frequency",main="Samples using PT: mu=7")
truehist(as.vector(mixNormPT2LONG[[3]][,1001:10000]),nbins=40,xlab="x",ylab="Frequency",main="Samples using PT: mu=10")
dev.off()


