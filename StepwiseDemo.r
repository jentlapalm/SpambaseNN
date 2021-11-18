
Data <- read.table("SeaWatch C data.csv",header=T,sep=",")

Data$RealGross <- Data$GROSS*min(Data$CPI)/Data$CPI
Data$Rate <- Data$RealGross/Data$CNVHRS

Data$ReaganP <- Data$REAG/(Data$REAG+Data$CART+Data$ANDR)
Data$AnderP <- Data$ANDR/(Data$REAG+Data$CART+Data$ANDR)
Data$CarterP <- Data$CART/(Data$REAG+Data$CART+Data$ANDR)


# Variable Transformations ------------------------------------------------

qqnorm(Data$Rate)

qqnorm(Data$CNVHRS)
qqnorm(log(Data$CNVHRS)) # Use Log
qqnorm((Data$CNVHRS)^(1/3))
qqnorm((Data$CNVHRS)^(1/2))

qqnorm(Data$CPI)        # Use Raw
qqnorm(log(Data$CPI))

qqnorm(Data$POP80)
qqnorm(log(Data$POP80)) #Use log

qqnorm(Data$HHMEDI)
qqnorm(log(Data$HHMEDI)) # Use log

qqnorm(Data$PERCAPI)
qqnorm(log(Data$PERCAPI)) # Use log

qqnorm(Data$POVPR)
logit <- function(p) {
  return(log(p/(1-p)))
}
qqnorm(logit(Data$POVPR/100)) # Use logit

qqnorm(Data$COLLPR)
qqnorm(logit(Data$COLLPR/100)) # Use logit

qqnorm(Data$MAGE)
qqnorm(log(Data$MAGE)) #Use log
range(Data$MAGE,na.rm=T)
qqnorm(logit(Data$MAGE/100))

qqnorm(Data$ReaganP)        # Use Raw
qqnorm(logit(Data$ReaganP))

qqnorm(Data$CarterP)        # Use Raw
qqnorm(logit(Data$CarterP)) # Slightly better, but use raw


qqnorm(Data$AnderP)        # Use Raw
qqnorm(logit(Data$AnderP)) # Better, but has huge outliers

qqnorm(Data$RealGross)
qqnorm(log(Data$RealGross)) #Use Log
qqnorm((Data$RealGross)^(1/3))


# Run Stepwise ------------------------------------------------------------

SmallFm <- log(RealGross) ~ 1
BigFm <- log(RealGross) ~ 1 + log(CNVHRS) + MON + VISIT + LST + CPI + log(POP80) +
  log(HHMEDI) + log(PERCAPI) + logit(POVPR/100) + logit(COLLPR/100) + log(MAGE) + 
  ReaganP + CarterP + AnderP

DataComp <- Data[complete.cases(Data),]

OutBig <- lm(BigFm,data=DataComp)

OutSmall <- lm(SmallFm,data=DataComp)
summary(OutSmall)

sc <- list(lower=SmallFm,upper=BigFm)
out <- step(OutSmall,scope=sc,direction="both")
summary(out)
AIC(out)

#save(DataComp,file="DataComp.Rdata")

out <- step(OutBig,scope=sc,direction="both")
summary(out)
AIC(out)
