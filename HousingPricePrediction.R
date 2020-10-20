install.packages("moments")
install.packages("MASS")
install.packages("glmnet")
install.packages("Metrics")
install.packages("doMC", repos="http://R-Forge.R-project.org")
install.packages("iterators")
install.packages("parallel")
install.packages("xgboost")
install.packages("caret")
install.packages("ISLR")
install.packages("neuralnet")
install.packages("pROC")
install.packages("gains")
library(gains)
library(pROC)
library(neuralnet)
library(ISLR)
library(caret)
library(xgboost)
library(iterators)
library(parallel)
library(doMC)
library(glmnet)
library(Metrics)
library(moments)
library(MASS)


setwd("C:/Users/Arjun/Documents/R")

train=read.csv("train.csv",stringsAsFactors = FALSE)
test=read.csv("test.csv",stringsAsFactors = FALSE)

dim(train)
dim(test)
str(train)

## Data pre-processing

train_ID=train$Id
test_ID=test$Id

test$SalePrice=NA

#Removing outliers

library(ggplot2)
dim(train)

# Living Area
qplot(train$GrLivArea,train$SalePrice,main="With Outliers")
train<-train[-which(train$GrLivArea>4000 & train$SalePrice<300000),]
qplot(train$GrLivArea,train$SalePrice,main="Without Outliers")

#LotShape
LotShape = train[["LotShape"]]
lsdf = data.frame(matrix(unlist(LotShape), nrow = length(LotShape), byrow = T), stringsAsFactors = FALSE)
train$LotShape=factor(train$LotShape,levels = c('Reg','IR1','IR2','IR3'), labels = c(1,2,3,4))
plot(train$SalePrice~train$LotShape, pch=16,cex=0.6)
i = which(train$LotShape == 4 & train$SalePrice < 75000)
train = train[-which(train$LotShape == 4 & train$SalePrice < 75000),]
lsdf = lsdf[-i,]
plot(train$SalePrice~train$LotShape, pch=16,cex=0.6)
train = train[, -match("LotShape", names(train))]
train = cbind(train,LotShape = lsdf)

dim(train)

#Foundation
Foundation = train[["Foundation"]]
fdf = data.frame(matrix(unlist(Foundation), nrow = length(Foundation), byrow = T), stringsAsFactors = FALSE)
train$Foundation=factor(train$Foundation,levels = c("BrkTil","CBlock","PConc","Slab","Stone","Wood"),labels = c(1,2,3,4,5,6))
plot(train$SalePrice~train$Foundation,pch=16,cex=0.6)
i=which(train$Foundation==1 & train$SalePrice > 4e+05)
train<-train[-which(train$Foundation == 1 & train$SalePrice > 4e+05),]
fdf = fdf[-i,]
plot(train$SalePrice~train$Foundation,pch=16,cex=0.6)
train = train[, -match("Foundation", names(train))]
train = cbind(train, Foundation = fdf)

dim(train)

#Log Transformation of SalePrice Variable
## Plot histogram of SalePrice Variable - Right skewed
qplot(SalePrice,data=train,bins=50,main="Right skewed distribution")

## Log transformation of the target variable
train$SalePrice <- log(train$SalePrice + 1)

## Normal distribution after transformation
qplot(SalePrice,data=train,bins=50,main="Normal distribution after log transformation")

train = train[c("Id","MSSubClass","MSZoning","LotFrontage","LotArea","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","Heating","HeatingQC","CentralAir","Electrical","X1stFlrSF","X2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF","EnclosedPorch","X3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal","MoSold","YrSold","SaleType","SaleCondition","SalePrice")]

dim(train)
dim(test)

## combine train and test
combine=rbind(train,test)

## Dropping Id as it is unnecessary for the prediction process.
combine=combine[,-1]

# Data Processing and Analysis

colSums(is.na(combine))

## Imputing Missing data
## For some variables, fill NA with "None" 
for(x in c("Alley","PoolQC","MiscFeature","Fence","FireplaceQu","GarageType","GarageFinish","GarageQual",'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"MasVnrType")){
  combine[is.na(combine[,x]),x]="None"
}

# factoring Neighborhood and Land contour
Neighborhood = combine[["Neighborhood"]]
LandContour = combine[["LandContour"]]

nhdf = data.frame(matrix(unlist(Neighborhood), nrow = length(Neighborhood), byrow = T), stringsAsFactors = FALSE)
lcdf = data.frame(matrix(unlist(LandContour), nrow = length(LandContour), byrow = T), stringsAsFactors = FALSE)

combine$Neighborhood=factor(combine$Neighborhood,levels = c("Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr","Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","NAmes","NoRidge","NPkVill","NridgHt","NWAmes","OldTown","SWISU","Sawyer","SawyerW","Somerst","StoneBr","Timber","Veenker"),labels = c("Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr","Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","NAmes","NoRidge","NPkVill","NridgHt","NWAmes","OldTown","SWISU","Sawyer","SawyerW","Somerst","StoneBr","Timber","Veenker"))
combine$LandContour=factor(combine$LandContour,levels = c('Lvl','Bnk','HLS','Low'), labels = c('Lvl','Bnk','HLS','Low'))

temp=aggregate(LotFrontage~LandContour+Neighborhood,data=combine,median)
dim(temp)

tempA = data.frame("Merged" = paste(temp$LandContour,temp$Neighborhood,sep='_'),"LotFrontage"=temp$LotFrontage)
dim(tempA)

tempdf = data.frame("Merged" = paste(combine$LandContour,combine$Neighborhood,sep = '_'), "LotFrontage" = combine$LotFrontage)
dim(tempdf)

index=which(is.na(tempdf$LotFrontage))

tempdf = tempdf[index,]
dim(tempdf)

tempdf

temp2 = c()

i = 0

for (str in tempdf$Merged) {
  i = i+1
  print(str)
  temp2 = c(temp2, which(tempA$Merged == str))
}

dim(tempA)
length(temp2)

combine$LotFrontage[is.na(combine$LotFrontage)] = temp[temp2, 3]

## Replacing missing data with 0
for(col in c('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',"MasVnrArea")){
  combine[is.na(combine[,col]),col]=0
}

## Replace missing MSZoning values by "RL" RL was majority
combine$MSZoning[is.na(combine$MSZoning)]="RL"

## Remove Utilities as it has zero variance 
combine=combine[, - match("Utilities", names(combine))]



## Replace missing Functional values with "Typ"
combine$Functional[is.na(combine$Functional)]="Typ"

## Replace missing Electrical values with "SBrkr"
combine$Electrical[is.na(combine$Electrical)]="SBrkr"

## Replace missing KitchenQual values by "TA"
combine$KitchenQual[is.na(combine$KitchenQual)]="TA"

## Replace missing SaleType values by "WD"
combine$SaleType[is.na(combine$SaleType)]="WD"

## Replace missing Exterior1st and Exterior2nd values by "VinylSd"
combine$Exterior1st[is.na(combine$Exterior1st)]="VinylSd"
combine$Exterior2nd[is.na(combine$Exterior2nd)]="VinylSd"

## All NAs should be gone, except the test portion of SalePrice variable, which we ourselves had initialized to NA earlier.
colSums(is.na(combine))
########################################

combine = combine[,-match("Neighborhood", names(combine))]
combine = combine[,-match("LandContour", names(combine))]

combine = cbind(combine, "Neighborhood" = nhdf)
combine = cbind(combine, "LandContour" = lcdf)

colnames(combine)[ncol(combine)-1] = "Neighborhood"
colnames(combine)[ncol(combine)] = "LandContour"

View(combine)

## Transforming some numerical variables that are really categorical

combine$MSSubClass=as.character(combine$MSSubClass)
combine$OverallCond=as.character(combine$OverallCond)
combine$YrSold=as.character(combine$YrSold)
combine$MoSold=as.character(combine$MoSold)


## Label Encoding some categorical variables that may contain information in their ordering set
  
cols = c('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope','LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold')

FireplaceQu=c('None','Po','Fa','TA','Gd','Ex')
BsmtQual=c('None','Po','Fa','TA','Gd','Ex')
BsmtCond=c('None','Po','Fa','TA','Gd','Ex')
GarageQual=c('None','Po','Fa','TA','Gd','Ex')
GarageCond=c('None','Po','Fa','TA','Gd','Ex')
ExterQual=c('Po','Fa','TA','Gd','Ex')
ExterCond=c('Po','Fa','TA','Gd','Ex')
HeatingQC=c('Po','Fa','TA','Gd','Ex')
PoolQC=c('None','Fa','TA','Gd','Ex')
KitchenQual=c('Po','Fa','TA','Gd','Ex')
BsmtFinType1=c('None','Unf','LwQ','Rec','BLQ','ALQ','GLQ')
BsmtFinType2=c('None','Unf','LwQ','Rec','BLQ','ALQ','GLQ')
Functional=c('Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ')
Fence=c('None','MnWw','GdWo','MnPrv','GdPrv')
BsmtExposure=c('None','No','Mn','Av','Gd')
GarageFinish=c('None','Unf','RFn','Fin')
LandSlope=c('Sev','Mod','Gtl')
LotShape=c('IR3','IR2','IR1','Reg')
PavedDrive=c('N','P','Y')
Street=c('Pave','Grvl')
Alley=c('None','Pave','Grvl')
MSSubClass=c('20','30','40','45','50','60','70','75','80','85','90','120','150','160','180','190')
OverallCond=NA
MoSold=NA
YrSold=NA
CentralAir=NA
levels=list(FireplaceQu, BsmtQual, BsmtCond, GarageQual, GarageCond, ExterQual, ExterCond,HeatingQC, PoolQC, KitchenQual, BsmtFinType1, BsmtFinType2, Functional, Fence, BsmtExposure, GarageFinish, LandSlope,LotShape, PavedDrive, Street, Alley, CentralAir, MSSubClass, OverallCond, YrSold, MoSold)
i=1
for (c in cols){
  if(c=='CentralAir'|c=='OverallCond'|c=='YrSold'|c=='MoSold'){
    combine[,c]=as.numeric(factor(combine[,c]))}
  else
    combine[,c]=as.numeric(factor(combine[,c],levels=levels[[i]]))
  i=i+1
}

combine$TotalSF=combine$TotalBsmtSF+combine$X1stFlrSF+combine$X2ndFlrSF


## Getting dummy categorical features


# first get data type for each feature
feature_classes <- sapply(names(combine),function(x){class(combine[[x]])})
numeric_feats <-names(feature_classes[feature_classes != "character"])

numeric_feats
# get names of categorical features
categorical_feats <- names(feature_classes[feature_classes == "character"])

# use caret dummyVars function for hot one encoding for categorical features
library(caret)
dummies <- dummyVars(~.,combine[categorical_feats])
categorical_1_hot <- predict(dummies,combine[categorical_feats])

## Fixing Skewed features

## We will transform the skewed features with BoxCox Transformation.

for(feat in numeric_feats){
  combine[[feat]] = as.numeric(combine[[feat]])
  print(class(combine[[feat]]))
}
## Determine skew for each numeric feature

skewed_feats <- sapply(numeric_feats,function(x){skewness(combine[[x]],na.rm=TRUE)})

#for (feat in numeric_feats) {
#  print(feat)
#  skewness(combine[[feat]])
#}

## Keep only features that exceed a threshold (0.75) for skewness
skewed_feats <- skewed_feats[abs(skewed_feats) > 0.75]

for(feat in names(skewed_feats)){
  print(feat)
}

names(skewed_feats)
## Transform skewed features with boxcox transformation

for(x in names(skewed_feats)) {
  #bc=BoxCoxTrans(combine[[x]],lambda = .15)
  #combine[[x]]=predict(bc,combine[[x]])
  if(!is.na(x)) {
  combine[[x]] <- log(combine[[x]] + 1)
  }
}



## Reconstruct all data with pre-processed data.


combine <- cbind(combine[numeric_feats],categorical_1_hot)

## Let us look at the dimensions of combine.
dim(combine)


# Model building and evaluation

## Splitting train dataset further into Training and Validation in order to evaluate the models
dim(combine)
combine$SalePrice[1457]

names(combine)
colnames(combine)[match("MSZoningC (all)", names(combine))] = "MSZoningC"

training<-combine[1:1456,]
testing<-combine[1457:2915,]
sampling = sample(1:nrow(training), 0.7*nrow(training))
Training<-training[sampling,]
Validation<-training[-sampling,]

###############################################


## Models
## Building Model on Validation set

## Lasso - Regularized Regression
set.seed(123)
lasso=cv.glmnet(as.matrix(Training[,-match("SalePrice", names(Training))]),Training[,match("SalePrice", names(Training))])
lasso
plot(lasso)
## Predictions
lasso_val_pr<-predict(lasso,newx=as.matrix(Validation[,-match("SalePrice", names(Training))]),s="lambda.min")
## RMSE Score
rmse_lasso = rmse(Validation$SalePrice,lasso_val_pr)

rmse_lasso

## GBM
set.seed(222)
## detectCores() returns 16 cpus
registerDoMC(16)
## Set up caret model training parameters
CARET.TRAIN.CTRL <-trainControl(method="repeatedcv",number=5,repeats=5,verboseIter=FALSE,allowParallel=TRUE)
gbm<-train(SalePrice~.,method="gbm",metric="RMSE",maximize=FALSE,trControl=CARET.TRAIN.CTRL,tuneGrid=expand.grid(n.trees=(4:10)*50,interaction.depth=c(5),shrinkage=c(0.05),n.minobsinnode=c(10)),data=Training,verbose=FALSE)
## Predictions
gbm_val_pr <- predict(gbm,newdata=Validation)
#RMSE Score
rmse_gbm = rmse(Validation$SalePrice,gbm_val_pr)
rmse_gbm
# Lift Chart
gain <- gains(Validation$SalePrice, gbm_val_pr)
gain
price = Validation$SalePrice
plot(c(0,gain$cume.pct.of.total*sum(price)/10000)~c(0,gain$cume.obs), type = 'l', col = 'blue', xlab = "Observation", ylab = "Gain", main = "GBM Lift Chart")
lines(c(0,sum(price)/10000)~c(0,dim(Validation)[1]), col = "gray", lty = 2)



## XGBOOST
install.packages("DiagrammeR")
library(DiagrammeR)
set.seed(123)
## Model parameters trained using xgb.cv function
xgb=xgboost(data=as.matrix(Training[,-match("SalePrice", names(Training))]),nfold=5,label=as.matrix(Training$SalePrice),nrounds=2200,verbose=FALSE,objective='reg:linear',eval_metric='rmse',nthread=8,eta=0.01,gamma=0.0468,max_depth=6,min_child_weight=1.7817,subsample=0.5213,colsample_bytree=0.4603)
xgb
## Predictions
xgb_val_pr <- predict(xgb,newdata=as.matrix(Validation[,-match("SalePrice", names(Training))]))
rmse_xgb = rmse(Validation$SalePrice,xgb_val_pr)
rmse_xgb
# XGB Lift Chart
gain <- gains(Validation$SalePrice, xgb_val_pr)
price = Validation$SalePrice
plot(c(0,gain$cume.pct.of.total*sum(price)/10000)~c(0,gain$cume.obs), type = 'l', col = 'red', xlab = "Observation", ylab = "Gain", main = "XGBoost Lift Chart")
lines(c(0,sum(price)/10000)~c(0,dim(Validation)[1]), col = "black", lty = 2)

# Normalization function
normalization <- function(dataframe) {
  
  for (col in colnames(dataframe)) {
    dataframe[,col] = (dataframe[,col] - min(dataframe[,col]))/(max(dataframe[,col])-min(dataframe[,col]))
  }
  return(dataframe)
}

normalization(Training)
normalization(Validation)

# K-Nearest Neighbors
ctrl <- trainControl(method="repeatedcv",repeats = 3)
knn = train(SalePrice ~ ., data = Training, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
plot(knn)
knn
knn_val_pr = predict(knn, Validation)
rmse_knn = rmse(Validation$SalePrice, knn_val_pr)

rmse_knn

## Lift chart for KNN

gain <- gains(Validation$SalePrice, knn_val_pr)
price = Validation$SalePrice
plot(c(0,gain$cume.pct.of.total*sum(price)/10000)~c(0,gain$cume.obs), type = 'l', col = 'orange', xlab = 'Observations', ylab = 'Gain', main = 'K-NN Lift Chart')
lines(c(0,sum(price)/10000)~c(0,dim(Validation)[1]), col = "gray", lty = 2)



## Neural Network
n <- names(Training)
p = "SalePrice~MSSubClass+LotFrontage+LotArea+Street+Alley+LotShape+LandSlope+OverallQual+OverallCond+YearBuilt+YearRemodAdd+MasVnrArea+ExterQual+ExterCond+Foundation+BsmtQual+BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+BsmtFinSF2+BsmtUnfSF+TotalBsmtSF+HeatingQC+CentralAir+X1stFlrSF+X2ndFlrSF+LowQualFinSF+GrLivArea+BsmtFullBath+BsmtHalfBath+FullBath+HalfBath+BedroomAbvGr+KitchenAbvGr+KitchenQual+TotRmsAbvGrd+Functional+Fireplaces+FireplaceQu+GarageYrBlt+GarageFinish+GarageCars+GarageArea+GarageQual+GarageCond+PavedDrive+WoodDeckSF+OpenPorchSF+EnclosedPorch+X3SsnPorch+ScreenPorch+PoolArea+PoolQC+Fence+MiscVal+MoSold+YrSold+SalePrice+TotalSF+MSZoningC+MSZoningFV+MSZoningRH+MSZoningRL+MSZoningRM+LotConfigCorner+LotConfigCulDSac+LotConfigFR2+LotConfigFR3+LotConfigInside+Condition1Artery+Condition1Feedr+Condition1Norm+Condition1PosA+Condition1PosN+Condition1RRAe+Condition1RRAn+Condition1RRNe+Condition1RRNn+Condition2Artery+Condition2Feedr+Condition2Norm+Condition2PosA+Condition2PosN+Condition2RRAe+Condition2RRAn+Condition2RRNn+BldgType1Fam+BldgType2fmCon+BldgTypeDuplex+BldgTypeTwnhs+BldgTypeTwnhsE+HouseStyle1.5Fin+HouseStyle1.5Unf+HouseStyle1Story+HouseStyle2.5Fin+HouseStyle2.5Unf+HouseStyle2Story+HouseStyleSFoyer+HouseStyleSLvl+RoofStyleFlat+RoofStyleGable+RoofStyleGambrel+RoofStyleHip+RoofStyleMansard+RoofStyleShed+RoofMatlCompShg+RoofMatlMembran+RoofMatlMetal+RoofMatlRoll+RoofMatlWdShake+RoofMatlWdShngl+Exterior1stAsbShng+Exterior1stAsphShn+Exterior1stBrkComm+Exterior1stBrkFace+Exterior1stCBlock+Exterior1stCemntBd+Exterior1stHdBoard+Exterior1stImStucc+Exterior1stMetalSd+Exterior1stPlywood+Exterior1stStone+Exterior1stStucco+Exterior1stVinylSd+Exterior1stWdShing+Exterior2ndAsbShng+Exterior2ndAsphShn+Exterior2ndBrkFace+Exterior2ndCBlock+Exterior2ndCmentBd+Exterior2ndHdBoard+Exterior2ndImStucc+Exterior2ndMetalSd+Exterior2ndOther+Exterior2ndPlywood+Exterior2ndStone+Exterior2ndStucco+Exterior2ndVinylSd+MasVnrTypeBrkCmn+MasVnrTypeBrkFace+MasVnrTypeNone+MasVnrTypeStone+HeatingFloor+HeatingGasA+HeatingGasW+HeatingGrav+HeatingOthW+HeatingWall+ElectricalFuseA+ElectricalFuseF+ElectricalFuseP+ElectricalMix+ElectricalSBrkr+GarageType2Types+GarageTypeAttchd+GarageTypeBasment+GarageTypeBuiltIn+GarageTypeCarPort+GarageTypeDetchd+GarageTypeNone+MiscFeatureGar2+MiscFeatureNone+MiscFeatureOthr+MiscFeatureShed+MiscFeatureTenC+SaleTypeCOD+SaleTypeCon+SaleTypeConLD+SaleTypeConLI+SaleTypeConLw+SaleTypeCWD+SaleTypeNew+SaleTypeOth+SaleTypeWD+SaleConditionAbnorml+SaleConditionAdjLand+SaleConditionAlloca+SaleConditionFamily+SaleConditionNormal+SaleConditionPartial+NeighborhoodBlmngtn+NeighborhoodBlueste+NeighborhoodBrDale+NeighborhoodBrkSide+NeighborhoodClearCr+NeighborhoodCollgCr+NeighborhoodCrawfor+NeighborhoodEdwards+NeighborhoodGilbert+NeighborhoodIDOTRR+NeighborhoodMeadowV+NeighborhoodMitchel+NeighborhoodNAmes+NeighborhoodNoRidge+NeighborhoodNPkVill+NeighborhoodNridgHt+NeighborhoodNWAmes+NeighborhoodOldTown+NeighborhoodSawyer+NeighborhoodSawyerW+NeighborhoodSomerst+NeighborhoodStoneBr+NeighborhoodSWISU+NeighborhoodTimber+NeighborhoodVeenker+LandContourBnk+LandContourHLS+LandContourLow+LandContourLvl"
f = as.formula(p)
nn = neuralnet(f, Training, hidden = c(5,5,5,5,5), threshold = 0.001, linear.output = TRUE, learningrate = 1e-5)
plot(nn, dimension = 10, arrow.length = 0.1)
nn_val_pr = compute(nn, Validation)
rmse_nn = rmse(Validation$SalePrice, nn_val_pr$net.result)
rmse_nn

RMSE = c(rmse_knn,rmse_xgb,rmse_gbm,rmse_lasso, rmse_nn)
names(RMSE) = c("K-NN", "XGBoost","GBM","Lasso","NeuralN")

barplot(RMSE, col = "red", ylab = 'RMSE Score', xlab = 'ML Model')




