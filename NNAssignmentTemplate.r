
set.seed(20201116)
DataOrig <- read.table("spambasedata-Orig.csv",sep=",",header=T,
                       stringsAsFactors=F)

ord <- sample(nrow(DataOrig))
DataOrig <- DataOrig[ord,]

# Change IsSpam to a factor

DataOrig$IsSpam <- factor(DataOrig$IsSpam)

# Doing a 60-20-20 split
TrainInd <- ceiling(nrow(DataOrig)*0.6)
TrainDF <- DataOrig[1:TrainInd,]
tmpDF <- DataOrig[-(1:TrainInd),]
ValInd <- ceiling(nrow(tmpDF)*0.5)
ValDF <- tmpDF[1:ValInd,]
TestDF <- tmpDF[-(1:ValInd),]

remove(TrainInd,tmpDF,ValInd,ord)

# Question 1 --------------------------------------------------------------
#
# Stepwise Logistic Regression

# I am setting up the formulas for you. You should examine how the BigFM
# is created, however.

SmallFm <- IsSpam ~ 1
Vars <- names(TrainDF)
BigFm <- paste(Vars[58],"~",paste(Vars[1:57],collapse=" + "),sep=" ")
BigFm <- formula(BigFm)

# Your code to do stepwise logistic regression and compute the predicted
# probabilities for the validation and test data goes here.

source("RocPlot.r")
ROCPlot(LRValP,ValDF$IsSpam)

# Question 2 --------------------------------------------------------------

if(!require("randomForest")) { install.packages("randomForest"); require("randomForest") }

# Your code to compute the random forest and compute the predicted
# probabilities for the validation and test data goes here.

ROCPlot(RFValP,ValDF$IsSpam)

# Question 3 Wide --------------------------------------------------------------

# Write out the data for the neural net models

write.table(TrainDF,file="NNHWTrain.csv",sep=",",row.names=F,col.names=T)
write.table(ValDF,file="NNHWVal.csv",sep=",",row.names=F,col.names=T)
write.table(TestDF,file="NNHWTest.csv",sep=",",row.names=F,col.names=T)

# -------------------------------------------------------------------------

# Read in the neural net output and compute the AUC for the validation data.
SpamNNWideTrainOutput <- read.table("SpamNNWideTrainDFOutput.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)
SpamNNWideValOutput <- read.table("SpamNNWideValDFOutput.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)
SpamNNWideTestOutput <- read.table("SpamNNWideTestDFOutput.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)

names(SpamNNWideValOutput)
ROCPlot(SpamNNWideValOutput$ValP,SpamNNWideValOutput$IsSpam)

# Question 3 Deep --------------------------------------------------------------
# 
# write.table(TrainDF,file="HWTrain.csv",sep=",",row.names=F,col.names=T)
# write.table(ValDF,file="HWVal.csv",sep=",",row.names=F,col.names=T)
# write.table(TestDF,file="HWTest.csv",sep=",",row.names=F,col.names=T)

# -------------------------------------------------------------------------

# Read in the neural net output and compute the AUC for the validation data.

SpamNNDeepTrainOutput <- read.table("SpamNNDeepTrainDFOutput.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)
SpamNNDeepValOutput <- read.table("SpamNNDeepValDFOutput.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)
SpamNNDeepTestOutput <- read.table("SpamNNDeepTestDFOutput.csv",header=T,sep=",",quote=NULL,stringsAsFactors = F)

names(SpamNNDeepValOutput)
ROCPlot(SpamNNDeepValOutput$ValP,SpamNNDeepValOutput$IsSpam)
