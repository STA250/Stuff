# NOTES:
# See: http://www.students.ncl.ac.uk/keith.newman/r/maps-in-r
# for a nice maps tutorial

library(maps)
library(mapdata)

foo <- read.csv("locresults.txt",header=TRUE)

heat.colpal <- heat.colors(n=diff(range(foo$sentiment))+2)
heat.colpal

"maptocolor" <- function(x,pal)
{
  return(pal[1+floor(x)-floor(min(x))])
}

pdf("tweetmap.pdf")
map('worldHires',c('USA','Canada'),xlim=c(-180,-50), ylim=c(20,80))
points(foo$lon,foo$lat,col=maptocolor(foo$sentiment,heat.colpal),pch=16)
dev.off()



