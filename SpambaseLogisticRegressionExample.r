
set.seed(20211111)
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

# Logistic Regression  --------------------------------------------------------------

names(TrainDF)

out <- glm(IsSpam ~ .,family=binomial(link = "logit"),data=TrainDF)
Probs <- predict(out,newdata=ValDF,type="response")

source("RocPlot.r")
ROCPlot(Probs,ValDF$IsSpam)

