# Configuration de l'environnement de travail et chargement des bibliothèques
setwd("/net/netud/m/imoussaid/S7/Data/Data/code/")
library(class)          # Pour k-NN
library(caret)          # Pour l'évaluation
library(ROCR)           # Pour ROC et AUC
library(FactoMineR)     # Pour ACP
library(corrplot)       # Pour visualisation des corrélations
library(MASS)           # Pour la régression
library(pls)            # Pour régression PLS
library(splines)        # Pour les splines dans les modèles de régression
library("FactoMineR")
library(PCAmixdata)
library(GGally)        # Pour la corelation

# Suppression des variables et graphiques
rm(list = ls(all = TRUE))
graphics.off()

# 1. Chargement des données d'entraînement et de test
load("../Projets/superconductivity_data_train.rda")
load("../Projets/superconductivity_data_test.rda")
data_train <- na.omit(data_train)  # Suppression des valeurs manquantes
data_test <- na.omit(data_test)


# 3. Prétraitement : Standardisation des données
# Z scoring

scale_it = function(x)
{
  out = (x - mean(x)/sd(x))
  out
}

scaled_train = data_train
for(i in 1:81)
{
  scaled_train[,i] = scale_it(scaled_train[,i])
}
# 
# 
# set.seed(123) # fixer une graine pour une reproductibilte
# sample_size <- floor(nrow(scaled_train) * 20/100)
# scaled_train <- scaled_train[sample(nrow(scaled_train), sample_size),]

# Modèle linéaire simple
linear_mod <- lm(critical_temp ~ scaled_train$std_atomic_mass ,data = scaled_train)

# Résumé du modèle
summary(linear_mod)

# Prédictions sur l'ensemble de test
predictions <- predict(linear_mod, newdata = data_test)

## visualisation
#1 
plot(linear_mod$fitted ~ scaled_train$critical_temp , pch = 19,
     ylab = "predicted critical temp (K)" , xlab = "observed critical temp (K)" )
abline(a = 0 , b = 1 , col = "red")
#2
plot(linear_mod$residuals ~ scaled_train$critical_temp , pch = 19 , 
     ylab = "residuals" , xlab = "observed critical temp (K)")
abline(h = 0 , lty = 2)
#3
hist(linear_mod$residuals , col = 'grey' , freq = F , xlab = 'Residuals' , main = "")

# vaidation croisee
sample_random_num <- 5 # valeur min
sample_size <- floor(nrow(scaled_train) * 20/100)
mse <- c()

for(i in 1 : sample_random_num)
{
  index_for_test_data = sample(1:nrow(scaled_train) , size = sample_size)
  tmp_test = scaled_train[index_for_test_data , ]
  tmp_train = scaled_train[-index_for_test_data ,]
  tmp_model = lm(critical_temp ~ tmp_train$mean_atomic_mass , data = tmp_train)
  tmp_y = predict(tmp_model , tmp_test[ , 1:81])
  mse <- c(mse , mean((tmp_y - tmp_test[, 82])^2))
  print(c(i , mse[i]))
}

print("full mse")
print(mean(mse))















# Et le non linéaire ?
## Cas spline (simple = une variable)
degoffreedom <- 4
# simpleSplineReg <- lm(y~ns(x1,degoffreedom), data=data)
spline_mod <- lm(critical_temp ~ ns(mean_atomic_mass, degoffreedom), data = scaled_train)


# Intervalle de confiance des paramètres estimés
# Risque à 5 % avec l’intervalle de confiance
confint(spline_mod)

# Adéquation au modèle avec le R^2
summary(spline_mod)
# Affichage de l’intervalle de confiance et de prédiction
seqx1 <- seq(min(scaled_train$mean_atomic_mass),max(scaled_train$mean_atomic_mass),length=50)
intpred <- predict(spline_mod,data.frame(mean_atomic_mass=seqx1),
                   interval="prediction")[,c("lwr","upr")]

plot(scaled_train$critical_temp~scaled_train$mean_atomic_mass,xlab="x1",ylab="y")

predictions <- predict(spline_mod, newdata = data_train)
#1 
plot(spline_mod$fitted ~ scaled_train$critical_temp , pch = 19,
     ylab = "predicted critical temp (K)" , xlab = "observed critical temp (K)" )
abline(a = 0 , b = 1 , col = "red")

# Validation du modèle par validation croisée
mse_spline <- c()

for(i in 1 : sample_random_num)
{
  index_for_test_data = sample(1:nrow(scaled_train) , size = sample_size)
  tmp_test = scaled_train[index_for_test_data , ]
  tmp_train = scaled_train[-index_for_test_data ,]
  tmp_model =  lm(critical_temp ~ ns(mean_atomic_mass, degoffreedom), data = tmp_train)
  tmp_y = predict(tmp_model , tmp_test[ , 1:81])
  mse_spline <- c(mse_spline , mean((tmp_y - tmp_test[, 82])^2))
  print(c(i , mse_spline[i]))
}

print("full mse")
print(mean(mse_spline))










## Selection de variables (en utilisant TOUTES les variables)
fullLinearReg <- lm(critical_temp~., data=scaled_train)
# back <- step(fullLinearReg, direction="backward", trace = 1)
# formula(back)
# null <- lm(critical_temp ~ ., data=scaled_train)
# forw <- step(null, scope=list(lower=null,upper=fullLinearReg),
#              direction="forward", trace = 1)
# formula(forw)

summary(fullLinearReg)

predictions <- predict(fullLinearReg, newdata = data_train)
mse <- c()

for(i in 1 : sample_random_num)
{
  index_for_test_data = sample(1:nrow(scaled_train) , size = sample_size)
  tmp_test = scaled_train[index_for_test_data , ]
  tmp_train = scaled_train[-index_for_test_data ,]
  tmp_model = lm(critical_temp ~ . , data = tmp_train)
  tmp_y = predict(tmp_model , tmp_test[ , 1:81])
  mse <- c(mse , mean((tmp_y - tmp_test[, 82])^2))
  print(c(i , mse[i]))
}

print("full mse")
print(mean(mse))













# # 4. Modèle avec splines
# spline_mod <- lm(critical_temp ~ ns(mean_atomic_mass, df = 8), data = scaled_train)
# summary(spline_mod)
# 
# 
# 
# # Visualisation des résultats pour les splines
# # 1. Observé vs prédit
# plot(spline_mod$fitted.values ~ scaled_train$critical_temp, pch = 19, 
#      ylab = "Prédictions", xlab = "Valeurs observées")
# abline(a = 0, b = 1, col = "blue")
# 
# # 2. Résidus vs observés
# spline_residuals <- scaled_train$critical_temp - spline_mod$fitted.values
# plot(spline_residuals ~ scaled_train$critical_temp, pch = 19, 
#      ylab = "Résidus", xlab = "Valeurs observées")
# abline(h = 0, lty = 2)
# 
# # 3. Histogramme des résidus
# hist(spline_residuals, col = "grey", freq = FALSE, xlab = "Résidus", main = "")
# 
# # Calcul de la MSE pour les splines
# # mse_spline <- mean((spline_predictions - scaled_test$critical_temp)^2)
# # cat("MSE pour le modèle avec splines :", mse_spline, "\n")
# 
# mse_spline <- numeric(sample_random_num)
# 
# for (i in 1:sample_random_num) {
#   index_test <- sample(1:nrow(scaled_train), size = sample_size)
#   tmp_test <- scaled_train[index_test, ]
#   tmp_train <- scaled_train[-index_test, ]
#   tmp_model <- lm(critical_temp ~ ns(mean_atomic_mass, df = 8), data = scaled_train)
#   tmp_pred <- predict(tmp_model, tmp_test)
#   mse_spline[i] <- mean((tmp_pred - tmp_test$critical_temp)^2)
# }
# 
# cat("MSE moyen pour le modèle avec splines :", mean(mse_spline), "\n")




##### Troisieme methode #####
library(ranger)

#modele foret aleatoire
num_trees <- 100 
randForest = ranger(critical_temp ~ . , data = scaled_train, mtry = 10 , min.node.size = 1 , 
                    num.trees = num_trees , importance = "permutation")
# quelque reprensation des resultat 
#1
plot(randForest$predictions ~ scaled_train$critical_temp, pch = 19,
     ylab = "predicted critical temperature (K)" , xlab="Observed critical temperatures (K)" )
abline(a = 0 , b = 1 , col = "red")

#2
rf_residuals=scaled_train$critical_temp-randForest$predictions
plot(rf_residuals ~ scaled_train$critical_temp, pch = 19,
     ylab = "Residuals (Obs-Pred)", xlab = "Observed critical temperatures (K)")
abline(h=0 ,lty=2)
#3
hist(rf_residuals,   col = "grey", freq = FALSE,  xlab = "Residuals", main = "")
#c'est mieux ! maintenant c'est bcp bcp plus long...
#si on ajoute la validation croisée dessous ça devient vraiment long!

mse_rf <- c()
for(i in 1:sample_random_num)
{
  index_for_test_data = sample(1:nrow(scaled_train) , size = sample_size)
  tmp_test = scaled_train[index_for_test_data, ]
  tmp_train = scaled_train[-index_for_test_data, ]
  tmp_model = ranger(critical_temp ~ .,data = tmp_train , mtry = 10 , num.trees = num_trees , 
                      importance = "permutation" , min.node.size = 1) 
  tmp = predict(randForest , tmp_test[,1:81])
  tmp_y = tmp$predictions
  mse_rf <- c(mse_rf , mean((tmp_y - tmp_test[,82])^2))
  print(c(i,mse_rf[i]))
}


print("full mse")
print(mean(mse_rf))


#quatrieme approche 
#on peut utiliser les forets que sur les premieres composante
#ACP 
n<-5 #a choisir ce nombre 
tab_train_wth_critical_temp= scaled_train[,1:81]
ACP_train<-PCA(tab_train_wth_critical_temp,graph=FALSE,ncp=n)
#construire les ncp nouvelles composantes dans un data frame 
ACP_dim<-as.numeric(ACP_train$var$cor[,1])
ACP_scaled_train<-data.frame(apply(scaled_train*ACP_dim,1,sum))
colnames(ACP_scaled_train)<-c(paste(c("dim",1),collapse = ""))

for (i in 2:n)
{
  ACP_dim<-as.numeric(ACP_train$var$cor[,i])
  temp_df<-scale_it(apply(scaled_train*ACP_dim,1,sum))
  ACP_scaled_train<-cbind(ACP_scaled_train,temp_df)
  colnames(ACP_scaled_train)[i]<-c(paste(c("dim",i), collapse = ""))
}

ACP_scaled_train<- cbind(ACP_scaled_train,scaled_train$critical_temp)
colnames(ACP_scaled_train)[n+1]<-c("critical_temp")

nb_trees<-100
mtry<-min(n,10)
# tmp_model =         ranger(critical_temp ~ .,data = tmp_train , mtry = 10 , num.trees = num_trees , 
#                    importance = "permutation" , min.node.size = 1) 
randforest_after_ACP= ranger(critical_temp ~ ., data=ACP_scaled_train, mtry=mtry, min.node.size = 1,
                             num.trees= nb_trees,importance="permutation")


mse_ACPrf <- c()
for(i in 1:sample_random_num)
{
  index_for_test_data = sample(1:nrow(scaled_train) , size = sample_size)
  tmp_test = scaled_train[index_for_test_data, ]
  tmp_train = scaled_train[-index_for_test_data, ]
  tmp_model = ranger(critical_temp ~ .,data = ACP_scaled_train , mtry = mtry , num.trees = nb_trees , 
                     importance = "permutation" , min.node.size = 1) 
  tmp = predict(randForest , tmp_test[,1:81])
  tmp_y = tmp$predictions
  mse_ACPrf <- c(mse_ACPrf , mean((tmp_y - tmp_test[,82])^2))
  print(c(i,mse_ACPrf[i]))
}


cat("MSE pour le modèle ACP avec forêts aléatoires :", mean(mse_ACPrf), "\n")



threshold <- 5
scaled_train <- scaled_train[scaled_train$critical_temp > threshold, ]

# Analyse discriminante QUADRATIQUE
quad_disc_an <- qda(critical_temp~., data = scaled_train)
# Prédiction sur les données test
test_QDA_predict <- predict(quad_disc_an, newdata=data_test_x, type="class")

