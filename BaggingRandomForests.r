# (c) 2017, George S. Easton

# Install libraries

if(!require("tree")) { install.packages("tree"); require("tree") }
if(!require("ISLR")) { install.packages("ISLR"); require("ISLR") }
if(!require("rgl")) { install.packages("rgl"); require("rgl") }

RglWindowPosition <- c(680,33,1275,780)

# Make Grid ---------------------------------------------------------------

# Make the grid for the fit surfaces

x1 <- Auto$weight
x2 <- Auto$horsepower
n <- 40
x1g <- (0:(n-1))/(n-1)*(max(x1)-min(x1))+min(x1)
x1g <- matrix(rep(x1g,n),ncol=n)
x2g <- (0:(n-1))/(n-1)*(max(x2)-min(x2))+min(x2)
x2g <- matrix(rep(x2g,n),ncol=n,byrow=T)
NewDF <- data.frame(weight=as.vector(x1g),horsepower=as.vector(x2g))

# Compute Tree ------------------------------------------------------------

# Compute regression tree once

tc <- tree.control(nrow(Auto),minsize=2,mincut=1,mindev=0)
out <- tree(mpg ~ weight + horsepower,data=Auto,control=tc)
summary(out)$size

# Plot Tree ---------------------------------------------------------------

NNodes <- 50
out1 <- prune.tree(out,best=NNodes)
print(out1)
plot(out1)
text(out1)

ypred <- predict(out1,newdata=NewDF)
ypred <- matrix(ypred,nrow=n)

par3d(windowRect=RglWindowPosition); rgl.bringtotop(stay=T); rgl.clear() # Positions the rgl window on my computer.
tmp1 <- plot3d(Auto$weight,Auto$horsepower,Auto$mpg,type="p",box=T,axes=T,size=5)
persp3d(x1g,x2g,ypred,front="lines",back="lines",add=T)

# Spin --------------------------------------------------------------------

play3d(spin3d(axis=c(0,0,1),rpm=3))

# Bagging -----------------------------------------------------------------

if(!require("randomForest")) { install.packages("randomForest"); require("randomForest") }

# When mtry = #of x's, randomForest is the same as bagging.

out2 <- randomForest(mpg ~ weight + horsepower,data=Auto,
                    mtry=2,ntree=5000,maxnodes=100)

ypred <- predict(out2,newdata=NewDF)
ypred <- matrix(ypred,nrow=n)

par3d(windowRect=RglWindowPosition); rgl.bringtotop(stay=T); rgl.clear() # Positions the rgl window on my computer.
tmp1 <- plot3d(Auto$weight,Auto$horsepower,Auto$mpg,type="p",box=T,axes=T,size=5)
persp3d(x1g,x2g,ypred,front="lines",back="lines",add=T)

play3d(spin3d(axis=c(0,0,1),rpm=3))


# Random Forest -----------------------------------------------------------

if(!require("randomForest")) { install.packages("randomForest"); require("randomForest") }

out2 <- randomForest(mpg ~ weight + horsepower,data=Auto,
                     mtry=1,ntree=5000,maxnodes=50)

ypred <- predict(out2,newdata=NewDF)
ypred <- matrix(ypred,nrow=n)

tmp1 <- plot3d(Auto$weight,Auto$horsepower,Auto$mpg,type="p",box=T,axes=T,size=5)
rgl.bringtotop(stay=T)
par3d(windowRect=c(800,33,1500,800)) # Places the rgl window on my computer.

persp3d(x1g,x2g,ypred,front="lines",back="lines",add=T)
#view3d(theta = 0,phi=0)

play3d(spin3d(axis=c(0,0,1),rpm=3))


# Comparison for Auto Data ------------------------------------------------

ind <- sample(nrow(Auto),size=100)
AutoVal <- Auto[ind,]
AutoTrain <- Auto[-ind,]

# Fit the full tree once
fm <- mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin     
tc <- tree.control(nrow(AutoTrain),minsize=2,mincut=1,mindev=0)
out <- tree(fm,data=AutoTrain,control=tc)
# get the # of end nodes
NNodes <- summary(out)$size
NNodes

# Find the best tree size
MSE.tree <- rep(NA,NNodes)
for(NNodes in 3:NNodes) {
  out1 <- prune.tree(out,best=NNodes)
  ypred <- predict(out1,newdata=AutoVal)
  MSE.tree[NNodes] <- mean((ypred-AutoVal$mpg)^2)
}
plot(MSE.tree,type="l")
BestN <- which.min(MSE.tree)
abline(v=BestN)
min(MSE.tree,na.rm=T)
MSE.tree[BestN]

out2 <- randomForest(fm,data=AutoTrain,
                     mtry=7,ntree=500,maxnodes=NNodes)
ypred <- predict(out2,newdata=AutoVal)
MSE.bag <- mean((ypred-AutoVal$mpg)^2)
MSE.bag

#What is the best value for Maxnodes
out3 <- randomForest(fm,data=AutoTrain,
                     ntree=500,maxnodes=NNodes)
ypred <- predict(out3,newdata=AutoVal)
MSE.rfor <- mean((ypred-AutoVal$mpg)^2)
MSE.rfor

c(MSE.tree[BestN],MSE.bag,MSE.rfor)

