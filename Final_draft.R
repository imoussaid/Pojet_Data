# Configuration de l'environnement de travail et chargement des bibliothèques nécessaires
# On commence par définir le répertoire de travail et charger les bibliothèques utiles pour les analyses.
setwd("/net/netud/m/imoussaid/S7/Data/Data/code/")
library(class)          # Pour k-NN (k-nearest neighbors)
library(caret)          # Pour évaluation des modèles et validation croisée
library(ROCR)           # Pour construire des courbes ROC et calculer l'AUC
library(FactoMineR)     # Pour réaliser une analyse en composantes principales (ACP)
library(corrplot)       # Pour visualiser les matrices de corrélation
library(MASS)           # Pour la régression et autres analyses statistiques
library(pls)            # Pour la régression PLS (Partial Least Squares)
library(splines)        # Pour les splines dans les modèles de régression
library("FactoMineR")   # Chargement redondant (possiblement à supprimer)
library(PCAmixdata)     # Pour ACP mixte (variables continues et catégoriques)
library(GGally)         # Pour des visualisations des corrélations
library(ranger)         # Pour l'algorithme de Random Forest

# Suppression des variables et graphiques
rm(list = ls(all = TRUE))
graphics.off()

# --- Chargement des données d'entraînement et de test ---
load("../Projets/superconductivity_data_train.rda")
load("../Projets/superconductivity_data_test.rda")
data_train <- na.omit(data_train)  # Suppression des lignes contenant des valeurs manquantes (NA)
data_test <- na.omit(data_test)


# --- Prétraitement des données : Standardisation ---
# Fonction pour effectuer une standardisation (z-score)
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
  
  # --- Entraînement et évaluation des modèles ---
# --- Modèle linéaire simple ---
linear_mod <- lm(critical_temp ~scaled_train$std_atomic_mass ,data = scaled_train)

# Visualisation des résultats du modèle linéaire
# Prédictions vs observations
plot(linear_mod$fitted ~ scaled_train$critical_temp , pch = 19,
     ylab = "predicted critical temp (K)" , xlab = "observed critical temp (K)" )
abline(a = 0 , b = 1 , col = "red")
# Résidus vs observations
plot(linear_mod$residuals ~ scaled_train$critical_temp , pch = 19 , 
     ylab = "residuals" , xlab = "observed critical temp (K)")
abline(h = 0 , lty = 2)
# Histogramme des résidus
hist(linear_mod$residuals , col = 'grey' , freq = F , xlab = 'Residuals' , main = "")


# Prédictions sur l'ensemble de test
predictions_linear <- predict(linear_mod, newdata = scaled_test)
# Affichage des premières prédictions
head(predictions_linear)

# --- Validation croisée pour évaluer le modèle linéaire ---
# Boucle pour effectuer une validation croisée avec 20% des données pour le test
  sample_random_num <- 5 # valeur min
  sample_size <- floor(nrow(scaled_train) * 20/100)
  mse <- c()
  for (i in 1:sample_random_num) {
    index_for_test_data = sample(1:nrow(scaled_train), size = sample_size)
    tmp_test = scaled_train[index_for_test_data, ]
    tmp_train = scaled_train[-index_for_test_data, ]
    
    tmp_model = lm(critical_temp ~ mean_atomic_mass, data = tmp_train)
    
    tmp_y = predict(tmp_model, newdata = tmp_test)
    
    mse <- c(mse, mean((tmp_y - tmp_test$critical_temp)^2))
    
    print(c(i, mse[i]))
  }
  
  cat("MSE pour le modèle lineaire simple :", mean(mse), "\n")

  ##### MSE élevée comment améliorer cette erreur ??

  
  #--------------------------- Modèle avec Spline--------------------------------------   
# Entraînement d'un modèle de régression avec splines naturelles
degoffreedom <- 4
spline_mod <- lm(critical_temp ~ ns(mean_atomic_mass, degoffreedom), data = scaled_train)

summary(spline_mod)


# Intervalle de confiance des paramètres estimés
# Risque à 5 % avec l’intervalle de confiance
confint(spline_mod)

# Visualisation des résultats du modèle avec spline
# Prédictions vs observations
plot(spline_mod$fitted ~ scaled_train$critical_temp , pch = 19,
     ylab = "predicted critical temp (K)" , xlab = "observed critical temp (K)" )
abline(a = 0 , b = 1 , col = "red")
# Résidus vs observations
plot(spline_mod$residuals ~ scaled_train$critical_temp , pch = 19 , 
     ylab = "residuals" , xlab = "observed critical temp (K)")
abline(h = 0 , lty = 2)
# Histogramme des résidus
hist(spline_mod$residuals , col = 'grey' , freq = F , xlab = 'Residuals' , main = "")
  
  
  # Prédictions sur l'ensemble de test
  predictions_spline <- predict(spline_mod, newdata = scaled_test)
  # Affichage des premières prédictions
  head(predictions_spline)

  # --- Validation croisée pour évaluer le modèle avec spline ---
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

  
  #--------------------------- Modèle Multivariables--------------------------------------   
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

# Prédictions sur l'ensemble de test
predictions_mlt <- predict(fullLinearReg, newdata = scaled_test)
head(predictions_mlt)

# --- Validation croisée pour évaluer le modèle en utilisant toutes les variables ---
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



#--------------------------- Modèle foret aleatoire-------------------------------------- 


start_time <- Sys.time()  # Temps de début

num_trees <- 81 
randForest = ranger(critical_temp ~ . , data = scaled_train, mtry = num_trees , min.node.size = 1 , 
                    num.trees = num_trees , importance = "permutation")
# Représentation des résultats 
# Prédictions vs observations

plot(randForest$predictions ~ scaled_train$critical_temp, pch = 19,
     ylab = "predicted critical temperature (K)" , xlab="Observed critical temperatures (K)" )
abline(a = 0 , b = 1 , col = "red")

# Résidus vs observations

rf_residuals=scaled_train$critical_temp-randForest$predictions
plot(rf_residuals ~ scaled_train$critical_temp, pch = 19,
     ylab = "Residuals (Obs-Pred)", xlab = "Observed critical temperatures (K)")
abline(h=0 ,lty=2)
# Histogramme des résidus

hist(rf_residuals,   col = "grey", freq = FALSE,  xlab = "Residuals", main = "")

# Faire des prédictions
predictions_rf <- predict(randForest, data = scaled_test)$predictions

# Afficher les premières prédictions
head(predictions_rf)


# --- Validation croisée pour évaluer le modèle random forest ---
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

## mse interessante sauf que temps de calcul trop eleve (superieure à 1mn) que faire?




#--------------------------- Modèle foret aleatoire avec ACP -------------------------------------- 



start_time <- Sys.time()  # Temps de début

n<-5 
ACP_train<-PCA(scaled_train[,1:81],graph=FALSE,ncp=n)
# construction de l'ACP pour les données train 
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

# --- Validation croisée pour évaluer le modèle random forest avec ACP---
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

## augmentation de la mse ce qui est attendus sauf que net amélioration du temps de calcul (inferieure à 30s) 


######################Validation du choix ACP+randomForest ############################
## excecution du code pour creer un fichier .csv de critical_temp sur les données test 



ACP_test<-PCA(scaled_test,graph=FALSE,ncp=n)

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