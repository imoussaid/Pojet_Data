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
library(GGally)        # Pour la correlation
library(ranger)        # Pour Random forest


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
      out}
    
    scaled_train = data_train
    for(i in 1:81)
      {scaled_train[,i] = scale_it(scaled_train[,i])}
    
    scaled_test = data_test
    for(i in 1:81)
    { scaled_test[,i] = scale_it(scaled_test[,i]) }
    
######################Entrainement et choix du model ############################


  #--------------------- Modèle linéaire simple-------------------------------------- 
  
  linear_mod <- lm(critical_temp ~scaled_train$std_atomic_mass ,data = scaled_train)
  
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
  
  
  # Prédictions sur l'ensemble de test
  predictions_linear <- predict(linear_mod, newdata = scaled_test)
  
  # pseudo visualisation
  head(predictions_linear)
  
  # vaidation croisee
  sample_random_num <- 5 # valeur min
  sample_size <- floor(nrow(scaled_train) * 20/100)
  mse <- c()
  for (i in 1:sample_random_num) {
    # Randomly sample indices for test and training data
    index_for_test_data = sample(1:nrow(scaled_train), size = sample_size)
    tmp_test = scaled_train[index_for_test_data, ]
    tmp_train = scaled_train[-index_for_test_data, ]
    
    # Build linear regression model correctly
    tmp_model = lm(critical_temp ~ mean_atomic_mass, data = tmp_train)
    
    # Make predictions ensuring 'tmp_test' matches the model formula
    tmp_y = predict(tmp_model, newdata = tmp_test)
    
    # Compute MSE
    mse <- c(mse, mean((tmp_y - tmp_test$critical_temp)^2))
    
    # Print iteration and MSE
    print(c(i, mse[i]))
  }
  
  cat("MSE pour le modèle lineaire simple :", mean(mse), "\n")

  ##### MSE élevée comment améliorer cette erreur

  
  --------------------------- Modèle avec Spline--------------------------------------   

  
  degoffreedom <- 4
  spline_mod <- lm(critical_temp ~ ns(mean_atomic_mass, degoffreedom), data = scaled_train)
  
  
  # Intervalle de confiance des paramètres estimés
  # Risque à 5 % avec l’intervalle de confiance
  confint(spline_mod)
    
  #1 
  plot(spline_mod$fitted ~ scaled_train$critical_temp , pch = 19,
       ylab = "predicted critical temp (K)" , xlab = "observed critical temp (K)" )
  abline(a = 0 , b = 1 , col = "red")
  #2
  plot(spline_mod$residuals ~ scaled_train$critical_temp , pch = 19 , 
       ylab = "residuals" , xlab = "observed critical temp (K)")
  abline(h = 0 , lty = 2)
  #3
  hist(spline_mod$residuals , col = 'grey' , freq = F , xlab = 'Residuals' , main = "")
  
  
  summary(spline_mod)
 
  predictions_spline <- predict(spline_mod, newdata = scaled_test)

  head(predictions_spline)

  
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
  
  cat("MSE pour le modèle avec spline :", mean(mse_spline), "\n") 

# Selection de variables (en utilisant TOUTES les variables) "méthode backward"
fullLinearReg <- lm(critical_temp~., data=scaled_train)
back <- step(fullLinearReg, direction="backward", trace = 1)
formula(back)

# On remarque que notre modèle prend toutes les variables en consideration
# auucne variable n'est prépendairante dans la prédiction de critical temp 

# Selection de variables (en utilisant TOUTES les variables) "méthode forward"

# null <- lm(critical_temp ~ ., data=scaled_train)
# forw <- step(null, scope=list(lower=null,upper=fullLinearReg),
#              direction="forward", trace = 1)
# formula(forw)

summary(fullLinearReg)

predictions_mlt <- predict(fullLinearReg, newdata = scaled_test)
head(predictions_mlt)
mse_mlt <- c()

for(i in 1 : sample_random_num)
{
  index_for_test_data = sample(1:nrow(scaled_train) , size = sample_size)
  tmp_test = scaled_train[index_for_test_data , ]
  tmp_train = scaled_train[-index_for_test_data ,]
  tmp_model = lm(critical_temp ~ . , data = tmp_train)
  tmp_y = predict(tmp_model , tmp_test[ , 1:81])
  mse_mlt <- c(mse_mlt , mean((tmp_y - tmp_test[, 82])^2))
  print(c(i , mse_mlt[i]))
}

cat("MSE pour apres une regression multiple:", mean(mse_mlt), "\n") 



--------------------------- Modèle foret aleatoire-------------------------------------- 


start_time <- Sys.time()  # Temps de début

num_trees <- 81 
randForest = ranger(critical_temp ~ . , data = scaled_train, mtry = num_trees , min.node.size = 1 , 
                    num.trees = num_trees , importance = "permutation")
# Représentation des résultats 
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

# Faire des prédictions
predictions_rf <- predict(randForest, data = scaled_test)$predictions

# Afficher les premières prédictions
head(predictions_rf)

# calcul de la mse
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



cat("MSE pour apres une regression multiple:", mean(mse_rf), "\n") 



end_time <- Sys.time()  


# Calcul du temps d'exécution
execution_time <- end_time - start_time
print(paste("Temps d'exécution pour random forest sans acp:", execution_time))




--------------------------- Modèle foret aleatoire avec ACP -------------------------------------- 


# Application de l'ACP 
start_time <- Sys.time()  # Temps de début

n<-5 #a choisir ce nombre 
scaled_train_without_critical_temp= scaled_train[,1:81]
ACP_train<-PCA(scaled_train_without_critical_temp,graph=FALSE,ncp=n)
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
randforest_after_ACP= ranger(critical_temp ~ ., data=ACP_scaled_train, mtry=mtry, min.node.size = 1,
                             num.trees= nb_trees,importance="permutation")

# Calcul de la mse
mse_ACPrf <- c()
for(i in 1:sample_random_num)
{
  index_for_test_data = sample(1:nrow(scaled_train) , size = sample_size)
  tmp_test = ACP_scaled_train[index_for_test_data, ]
  tmp_train = ACP_scaled_train[-index_for_test_data, ]
  tmp_model = ranger(critical_temp ~ .,data = ACP_scaled_train , mtry = mtry , num.trees = nb_trees , 
                     importance = "permutation" , min.node.size = 1) 
  tmp = predict(randforest_after_ACP , tmp_test[,1:n])
  tmp_y = tmp$predictions
  mse_ACPrf <- c(mse_ACPrf , mean((tmp_y - tmp_test[,n+1])^2))
  print(c(i,mse_ACPrf[i]))
}
cat("MSE pour le modèle ACP avec forêts aléatoires :", mean(mse_ACPrf), "\n")
end_time <- Sys.time()  

# Calcul du temps d'exécution
execution_time <- end_time - start_time

print(paste("Temps d'exécution pour randomforest after acp:", execution_time))


ACP_test<-PCA(scaled_test,graph=FALSE,ncp=n)
#construire les ncp nouvelles composantes dans un data frame 
ACP_dim<-as.numeric(ACP_test$var$cor[,1])
ACP_scaled_test<-data.frame(apply(scaled_test*ACP_dim,1,sum))
colnames(ACP_scaled_test)<-c(paste(c("dim",1),collapse = ""))

for (i in 2:n)
{
  ACP_dim<-as.numeric(ACP_test$var$cor[,i])
  temp_df<-scale_it(apply(scaled_test*ACP_dim,1,sum))
  ACP_scaled_test<-cbind(ACP_scaled_test,temp_df)
  colnames(ACP_scaled_test)[i]<-c(paste(c("dim",i), collapse = ""))
}
# Prédictions sur les données de test
predictions <- predict(randforest_after_ACP, ACP_scaled_test)
predicted_critical_temp <- predictions$predictions

head(predicted_critical_temp)

# Sauvegarde des prédictions dans un fichier
output <- data.frame(ID = 1:nrow(data_test), Predicted_Critical_Temp = predicted_critical_temp)
write.csv(output, "predicted_critical_temp.csv", row.names = FALSE)
cat("Les prédictions ont été sauvegardées dans 'predicted_critical_temp.csv'.\n")
