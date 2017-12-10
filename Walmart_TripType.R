# directorio de trabajo
setwd("C:/Users/lucky/Documents/Ciencia de Datos/Aprendizaje de Maquina/Proyecto Final ML")
set.seed(23456)

# Limpiamos la casa
rm(list=ls())
gc()

# cargamos las librerias necesarias para el proceso
library(readr)
library(tidyr)
library(dplyr)
library(nnet)
library(caret)
library(class)
library(xgboost)
library(randomForest)

# cargamos los datos
train <- read.csv('train/train.csv', header = T, stringsAsFactors = TRUE)
test <- read.csv('test/test.csv', header = T, stringsAsFactors = TRUE)

# Exploramos los datos y limpiamos
# Identify missing entries between train and test data sets and remove them
setdiff(unique(train$DepartmentDescription), unique(test$DepartmentDescription))

# Remove the Department that is NOT existing in Test data set
train <- train[! train$DepartmentDescription == "HEALTH AND BEAUTY AIDS", ]

## Preparacion de los datos


#Renombramos los targetTypes
TargetType <- unique(train$TripType)
targetdf <- data.frame(target = sort(TargetType), seqno = c(0 : (length(TargetType) -1)) )
train$target<-0
for (type in TargetType) {
  train[train$TripType == type,]$target <- targetdf[targetdf$target == type,]$seqno
}
train$TripType <- train$target
train$target <- NULL


# convertimos la base en una nueva, donde cada renglon corresponde a un viaje individual
# y cada columna son el numero de articulos comprados por departamento
visitas <- aggregate(train$ScanCount, by = list(train$VisitNumber, train$DepartmentDescription), FUN = sum)
visitas <- spread(visitas, Group.2, x, fill = 0)

# anadimos columnas con caracteristicas creadas por nosotros

#Variable total de productos scaneados
totProd <- train %>%
   dplyr::group_by(VisitNumber) %>%
   dplyr::summarise(prod = sum(ScanCount))
entrenamiento <- left_join(visitas, totProd, by = c("Group.1" = "VisitNumber"))

#Variable que indica si regresó algún producto
returnProd <- train%>%
  mutate(Return = ifelse(ScanCount <0,1,0))

returnProd <- returnProd %>%
    dplyr::group_by(VisitNumber)%>%
    dplyr::summarise(Return = sum(Return))

returnProd$Return <- ifelse(returnProd$Return >=1,1,0)

entrenamiento <- left_join(entrenamiento, returnProd, by = c("Group.1" = "VisitNumber"))

#Variable que indica el total de departamentos visitados durante el viaje
totDep <- train%>%
    dplyr::group_by(VisitNumber) %>%
    dplyr::summarise(totDep = n_distinct(DepartmentDescription))

entrenamiento <- left_join(entrenamiento, totDep, by = c("Group.1" = "VisitNumber"))

#Variable que indica el total de FineLineNumber comprados  durante el viaje
totFineLN <- train%>%
  dplyr::group_by(VisitNumber) %>%
  dplyr::summarise(totFineLN = n_distinct(FinelineNumber))

entrenamiento <- left_join(entrenamiento, totFineLN, by = c("Group.1" = "VisitNumber"))

#Variable que indica el total de UPC Products comprados  durante el viaje
totUPC <- train%>%
  dplyr::group_by(VisitNumber) %>%
  dplyr::summarise(totUpc = n_distinct(Upc))

entrenamiento <- left_join(entrenamiento, totUPC, by = c("Group.1" = "VisitNumber"))

# Generamos estadisticas: mean, max, min de productos comprados en cada viaje por depto
stats<-train%>%
  group_by(VisitNumber,DepartmentDescription)%>%
  summarise(prom = mean(ScanCount),
            min = min(ScanCount),
            max = max(ScanCount),
            range = (max -min))

spread_stats <- stats%>%
    select(VisitNumber, DepartmentDescription, prom)%>%
    spread(DepartmentDescription, prom,fill = 0)

names(spread_stats) <- paste(names(spread_stats),'prom') 

entrenamiento <- left_join(entrenamiento, spread_stats, by = c("Group.1" = "VisitNumber prom"))

spread_stats <- stats%>%
  select(VisitNumber, DepartmentDescription, min)%>%
  spread(DepartmentDescription, min,fill = 0)

names(spread_stats) <- paste(names(spread_stats),'min') 

entrenamiento <- left_join(entrenamiento, spread_stats, by = c("Group.1" = "VisitNumber min"))

spread_stats <- stats%>%
  select(VisitNumber, DepartmentDescription, max)%>%
  spread(DepartmentDescription, max,fill = 0)

names(spread_stats) <- paste(names(spread_stats),'max') 

entrenamiento <- left_join(entrenamiento, spread_stats, by = c("Group.1" = "VisitNumber max"))

spread_stats <- stats%>%
  select(VisitNumber, DepartmentDescription, range)%>%
  spread(DepartmentDescription, range,fill = 0)

names(spread_stats) <- paste(names(spread_stats),'range') 

entrenamiento <- left_join(entrenamiento, spread_stats, by = c("Group.1" = "VisitNumber range"))



# extraemos el dia de la semana y el tipo de viaje de cada viaje
caracteristicas <- unique(train[,c(2:3,1)])
# unimos ambas bases para completar nuestro conjunto de entrenamiento
entrenamiento <- left_join(entrenamiento, caracteristicas, by = c("Group.1" = "VisitNumber"))

#Categorizamos los dias de la semana y los tipos de viaje
entrenamiento$Weekday <- as.numeric(as.factor(entrenamiento$Weekday))
entrenamiento$TripType <- as.factor(entrenamiento$TripType)  



# Lo mismo que hicimos con train ahora va con test

# convertimos la base en una nueva, donde cada renglon corresponde a un viaje individual
# y cada columna son el numero de articulos comprados por departamento
visitas_test <- aggregate(test$ScanCount, by = list(test$VisitNumber, test$DepartmentDescription), FUN = sum)
visitas_test <- spread(visitas_test, Group.2, x, fill = 0)

# anadimos columnas con caracteristicas creadas por nosotros

#Variable total de productos scaneados
totProd <- test %>%
  dplyr::group_by(VisitNumber) %>%
  dplyr::summarise(prod = sum(ScanCount))
test_ds <- left_join(visitas_test, totProd, by = c("Group.1" = "VisitNumber"))

#Variable que indica si regresó algún producto
returnProd <- test%>%
  mutate(Return = ifelse(ScanCount <0,1,0))

returnProd <- returnProd %>%
  dplyr::group_by(VisitNumber)%>%
  dplyr::summarise(Return = sum(Return))

returnProd$Return <- ifelse(returnProd$Return >=1,1,0)

test_ds <- left_join(test_ds, returnProd, by = c("Group.1" = "VisitNumber"))

#Variable que indica el total de departamentos visitados durante el viaje
totDep <- test%>%
  dplyr::group_by(VisitNumber) %>%
  dplyr::summarise(totDep = n_distinct(DepartmentDescription))

test_ds <- left_join(test_ds, totDep, by = c("Group.1" = "VisitNumber"))

#Variable que indica el total de FineLineNumber comprados  durante el viaje
totFineLN <- test%>%
  dplyr::group_by(VisitNumber) %>%
  dplyr::summarise(totFineLN = n_distinct(FinelineNumber))

test_ds <- left_join(test_ds, totFineLN, by = c("Group.1" = "VisitNumber"))

#Variable que indica el total de UPC Products comprados  durante el viaje
totUPC <- test%>%
  dplyr::group_by(VisitNumber) %>%
  dplyr::summarise(totUpc = n_distinct(Upc))

test_ds <- left_join(test_ds, totUPC, by = c("Group.1" = "VisitNumber"))

# Generamos estadisticas: mean, max, min de productos comprados en cada viaje por depto
stats<-test%>%
  group_by(VisitNumber,DepartmentDescription)%>%
  summarise(prom = mean(ScanCount),
            min = min(ScanCount),
            max = max(ScanCount),
            range = (max -min))

spread_stats <- stats%>%
  select(VisitNumber, DepartmentDescription, prom)%>%
  spread(DepartmentDescription, prom,fill = 0)

names(spread_stats) <- paste(names(spread_stats),'prom') 

test_ds<- left_join(test_ds, spread_stats, by = c("Group.1" = "VisitNumber prom"))

spread_stats <- stats%>%
  select(VisitNumber, DepartmentDescription, min)%>%
  spread(DepartmentDescription, min,fill = 0)

names(spread_stats) <- paste(names(spread_stats),'min') 

test_ds <- left_join(test_ds, spread_stats, by = c("Group.1" = "VisitNumber min"))

spread_stats <- stats%>%
  select(VisitNumber, DepartmentDescription, max)%>%
  spread(DepartmentDescription, max,fill = 0)

names(spread_stats) <- paste(names(spread_stats),'max') 

test_ds <- left_join(test_ds, spread_stats, by = c("Group.1" = "VisitNumber max"))

spread_stats <- stats%>%
  select(VisitNumber, DepartmentDescription, range)%>%
  spread(DepartmentDescription, range,fill = 0)

names(spread_stats) <- paste(names(spread_stats),'range') 

test_ds <- left_join(test_ds, spread_stats, by = c("Group.1" = "VisitNumber range"))



# extraemos el dia de la semana y el tipo de viaje de cada viaje
caracteristicas_test <- unique(test[,1:2])
# unimos ambas bases para completar nuestro conjunto de entrenamiento
test_ds <- left_join(test_ds, caracteristicas_test, by = c("Group.1" = "VisitNumber"))
#Categorizamos los dias de la semana y los tipos de viaje
test_ds$Weekday <- as.numeric(as.factor(test_ds$Weekday))

# quitar los valores na
entrenamiento <-na.omit(entrenamiento)
test_ds <- na.omit(test_ds)

## Entrenamiento y seleccion de modelos

# knn
# Dividimos el set en train y test
randoms<-sample(1:100,95674, replace = TRUE)
randoms <- randoms/10000000
entrenamiento1 <- entrenamiento[,-1]
train1 <- entrenamiento1[,-ncol(entrenamiento1)]
train1_labels <- entrenamiento1[,ncol(entrenamiento1)]
train1 <- data.frame(train1,random = randoms)
#train1 <- entrenamiento1[1:76539,-ncol(entrenamiento1)]
randoms<-sample(1:100,95674, replace = TRUE)
randoms <- randoms/10000000
test1 <- test_ds[,-1]
test1<- data.frame(test1, random = randoms)
#test1<- entrenamiento1[76539:95674,-ncol(entrenamiento1)]
#train1_labels <- entrenamiento1[1:76539,ncol(entrenamiento1)]
#test1_labels <- entrenamiento1[76539:95674,ncol(entrenamiento1)]

# Entrenamos el modelo KNN con K = 1 porque si le ponemos mas no jala
test_pred <- knn(train = train1, test = test1,cl = train1_labels, k=5, l=0 ,use.all = FALSE)

# Vemos la eficacia del modelo
#library(gmodels)
#CrossTable(x=test1_labels, y = test_pred, prop.chisq =FALSE)
#subt2 <- data.frame(test_pred, test1_labels)
#corr2 <- diag(table(subt2))
#effi2 <- sum(corr2)
#effi2/nrow(test1)

# Tenemos un porcentaje de eficacia mayor al 58%... Chingon!
# Ahora armamos el archivo de respuesta

# Sacamos el numero de clases (Cada clase es una columna)
noOfClasses <- length(unique(train1_labels))
# Armamos una matriz con el VisitNumber, las predicciones, y una columna de 1's
preds<-data.frame(VisitNumber = test_ds[,1], Pred = test_pred, value = 1)
# Hacemos el spread para separar los TripTypes en columnas
preds <- spread(preds, Pred, value, fill = 0)
#Creamos un vector con los nombres finales de las columnas
preds2 <- data.frame(preds[,1:9], value = 0)
preds <- data.frame(preds2, preds[,10:ncol(preds)])
#target2<-targetdf[-9,]
nombres <- c("VisitNumber",paste("TripType_", targetdf$target, sep=""))
#Asignamos los nombres a las columnas
names(preds) <- nombres
#Escribimos el archivo
write.csv(format(preds, scientific = FALSE), '1-Knn_final.csv', row.names = F, quote = F)


# Tiramos un XGBOOST
entrenamiento1 <- entrenamiento[,-1]
train1_labels <- entrenamiento1[,ncol(entrenamiento1)]
train1_labels <- as.numeric(as.character(train1_labels))
noOfClasses <- length(unique(train1_labels))
param <- list('objective' = 'multi:softprob',
              'eval_metric' = 'mlogloss',
              'num_class' = noOfClasses)

cv.round <- 50
cv.nfold <- 5

### Model Generation
# Perform Cross-validation using the above params and objects
randoms<-sample(1:100,95674, replace = TRUE)
randoms <- randoms/10000000
train1 <- entrenamiento1[,-ncol(entrenamiento1)]
train1 <- data.frame(train1,random = randoms)
trainMatrix <- xgb.DMatrix(data = data.matrix(train1), label = train1_labels)

xgbcv <- xgb.cv(param = param, data = trainMatrix,
                label = train1_labels, nrounds = cv.round, 
                nfold = cv.nfold)
nround <- which(xgbcv$test.mlogloss.mean == min(xgbcv$test.mlogloss.mean) )

xgb_model <- xgboost(param = param, data = trainMatrix, label = train1_labels, nrounds = cv.round)

test1 <- test_ds[,-1]
test1 <- data.frame(test1, random = randoms)
testMatrix <- as.matrix(test1)
ypred <- predict(xgb_model, testMatrix)
predMatrix <- data.frame(matrix(ypred, byrow = TRUE, ncol = noOfClasses))
colnames(predMatrix) <- paste("TripType_", targetdf$target, sep="")
res <- data.frame(VisitNumber = test_ds[, 1], predMatrix)
result <- aggregate(. ~ VisitNumber, data = res, FUN = mean)
write.csv(format(result, scientific = FALSE), '2-XGBoost_final.csv', row.names = F, quote = F)

#Tiramos un RandomForest
#Cargamos los datos de entrenamiento sin el Id del registro
entrenamiento1 <- entrenamiento[,-1]
#Cargamos los datos de pruebas sin el id del registro
test_rf <- test_ds[,-1]
#Ponemos unos nombres Tidy, si no truena
names(entrenamiento1)[1]<-paste("PHOTO")
tidy.names <- make.names(nombres, unique = TRUE)
names(entrenamiento1) <- tidy.names
#Generamos el random forest
model<-randomForest(TripType ~ . ,data = entrenamiento1)
#Tidyficamos los nombres de pruebas tambien, quitemos la columna de salida
tidy.namestest <- tidy.names[-347]
names(test_rf) <- tidy.namestest
#Generamos la prediccion
output<-predict(model,test_rf)

#Armamos el archivo
# Armamos una matriz con el VisitNumber, las predicciones, y una columna de 1's
preds<-data.frame(VisitNumber = test_ds[,1], Pred = output, value = 1)
# Hacemos el spread para separar los TripTypes en columnas
preds <- spread(preds, Pred, value, fill = 0)
#Creamos un vector con los nombres finales de las columnas
preds2 <- data.frame(preds[,1:9], value = 0)
preds <- data.frame(preds2, preds[,10:ncol(preds)])
#target2<-targetdf[-9,]
nombres <- c("VisitNumber",paste("TripType_", target2$target, sep=""))
#Asignamos los nombres a las columnas
names(preds) <- nombres
#Escribimos el archivo
write.csv(format(preds, scientific = FALSE), '3-RandomForest_final.csv', row.names = F, quote = F)

