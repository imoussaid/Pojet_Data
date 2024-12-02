# Pojet_Data

# # # Adresse du dossier où vous travaillez
# # setwd("/net/netud/m/imoussaid/S7/Data/Data/code/")
# # # Packages utilisés dans la suite
# # library(class)
# # library(caret)
# # library(ROCR)
# # library(MASS)
# # # Supprimer toutes les variables
# # rm(list=ls(all=TRUE))
# # # Supprimer tous les graphiques déjà présents
# # graphics.off()
# #
# # load("../Projets/superconductivity_data_train.rda")
# # print(data_train)
# #
# #
# # load("../Projets/superconductivity_data_test.rda")
# #
# # head(data_train)
# # summary(data_train)
# # str(data_train)
# #
# # data_train <- na.omit(data_train)  # Pour supprimer les lignes avec des valeurs manquantes
# #
# # data_train$variable <- as.factor(data_train$variable)
# #
# # cor_matrix <- cor(data_train[, sapply(data_train, is.numeric)])
# # print(cor_matrix)
# #
# # library(corrplot)
# # corrplot(cor_matrix, method = "circle")
# #
# # acp <- prcomp(data_train[, sapply(data_train, is.numeric)], scale. = TRUE)
# # summary(acp)
# # plot(acp)
# ## Adresse du dossier où vous travaillez
# setwd("/net/netud/m/imoussaid/S7/Data/Data/code/")
# # Packages utilisés dans la suite
# library("FactoMineR")

# load("../Projets/superconductivity_data_train.rda")
# X <- data_train
# print(X)

# # Calcul de la moyenne et de l’écart type des variables
# mean <- apply(X,2,mean)
# std <- apply(X,2,sd) #standard deviation
# stat <- rbind(mean,std)

# # Affichage
# print(stat,digits=4)



# # Création des données centrées ...
# Xnorm <- sweep(X,2,mean,"-")
# # ... et réduites
# Xnorm <- sweep(Xnorm,2,std,"/")
# # Affichage des données centrées - réduites
# print(Xnorm,digits=4)



# # Nombre de clusters souhaité
# numcluster <- 5



# ## KMEANS
# # Algorithme des kmeans (avec affichage)
# km <- kmeans(X,numcluster,nstart=50)
# print(km)
# # Algorithme des kmeans sur données centrées-réduites (avec affichage)
# kmnorm <- kmeans(Xnorm,numcluster,nstart=50)
# print(kmnorm)





# # Concatenation des données avec leur résultat de cluster
# cluster <- as.factor(km$cluster)
# clusternorm <- as.factor(kmnorm$cluster)
# XplusCluster <- data.frame(X,cluster=cluster)
# XnormplusCluster <- data.frame(Xnorm,cluster=clusternorm)
# colclust <- length(X)+1
# print(XplusCluster)




# # ACP sur les données brutes
# rPCA <- PCA(XplusCluster,scale.unit=FALSE,graph=FALSE,quali.sup=colclust)

# # # Nuage des individus et des variables dans le premier plan factoriel
# # par(mfrow=c(1,2))
# # plot.PCA(rPCA,axes=c(1,2),choix="ind",habillage=colclust,invisible="quali")
# # plot.PCA(rPCA,axes=c(1,2),choix="var")
# # # Nuage des individus et des variables dans le deuxième plan factoriel
# # par(mfrow=c(1,2))
# # plot.PCA(rPCA,axes=c(1,3),choix="ind",habillage=colclust,invisible="quali")
# # plot.PCA(rPCA,axes=c(1,3),choix="var")




# #ACP sur les données centrées-réduites
# rPCAnorm <- PCA(XnormplusCluster,graph=FALSE,quali.sup=colclust)
# # Nuage des individus et des variables dans le premier plan factoriel
# par(mfrow=c(1,2))
# plot.PCA(rPCAnorm,,axes=c(1,2),choix="ind",habillage=colclust,invisible="quali",label = "none")
# plot.PCA(rPCAnorm,axes=c(1,2),choix="var")
# # Nuage des individus et des variables dans le deuxième plan factoriel
# par(mfrow=c(1,2))
# plot.PCA(rPCAnorm,axes=c(1,3),choix="ind",habillage=colclust,invisible="quali")
# plot.PCA(rPCAnorm,axes=c(1,3),choix="var")

# print(data_train$critical_temp)
# print(kmnorm$cluster)
# plot(kmnorm$cluster , data_train$critical_temp)

# data_train$cluster <- kmnorm$cluster

# cluster_stats <- function(cluster_data) {
#   mean_temp <- mean(cluster_data$critical_temp)
#   sd_temp <- sd(cluster_data$critical_temp)
#   return(c(mean = mean_temp, sd = sd_temp))
# }

# # Diviser les données par cluster
# clusters <- split(data_train, data_train$cluster)

# # Calculer la moyenne et l'écart-type pour chaque cluster
# stats_list <- lapply(clusters, cluster_stats)

# stats_table <- do.call(rbind, stats_list)

# print(stats_table)

# #
# #
#  # Pour supprimer les lignes avec des valeurs manquantes
# cor_matrix <- cor(data_train[, sapply(data_train, is.numeric)])
# print(cor_matrix)

# library(corrplot)
# corrplot(cor_matrix, method = "circle")


# #Classification hiérarchique de Ward sur données brutes
# d <- dist(X)
# tree <- hclust(d^2,method="ward.D2")
# par(mfrow=c(1,1))
# plot(tree)

# #Classification hiérarchique de Ward sur données centrées-réduites
# dnorm <- dist(Xnorm)
# treenorm <- hclust(dnorm^2,method="ward.D2")
# plot(treenorm)



# # Concatenation des données avec leur résultat de cluster
# clusterW <- as.factor(cutree(tree,numcluster))
# XplusClusterW <- data.frame(X,cluster=clusterW)
# print(XplusClusterW)
# clusternormW <- as.factor(cutree(treenorm,numcluster))
# XnormplusClustW <- data.frame(Xnorm,cluster=clusternormW)
# print(XnormplusClustW)



# # ACP sur les données brutes
# rPCAW <- PCA(XplusClusterW,scale.unit=FALSE,graph=FALSE,quali.sup=colclust)
# # Nuage des individus et des variables dans le premier plan factoriel
# par(mfrow=c(1,2))
# plot.PCA(rPCAW,axes=c(1,2),choix="ind",habillage=colclust,invisible="quali")
# plot.PCA(rPCAW,axes=c(1,2),choix="var")
# # Nuage des individus et des variables dans le deuxième plan factoriel
# par(mfrow=c(1,2))
# plot.PCA(rPCAW,axes=c(1,3),choix="ind",habillage=colclust,invisible="quali")
# plot.PCA(rPCAW,axes=c(1,3),choix="var")

# #
# #
# #
# #
# #ACP sur les données centrées-réduites
# rPCAnormW <- PCA(XnormplusClustW,scale.unit=FALSE,graph=FALSE,quali.sup=colclust)

# # Nuage des individus et des variables dans le premier plan factoriel
# par(mfrow=c(1,2))
# plot.PCA(rPCAnormW,axes=c(1,2),choix="ind",habillage=colclust,invisible="quali")
# plot.PCA(rPCAnormW,axes=c(1,2),choix="var")
# # Nuage des individus et des variables dans le deuxième plan factoriel
# par(mfrow=c(1,2))
# plot.PCA(rPCAnormW,axes=c(1,3),choix="ind",habillage=colclust,invisible="quali")
# plot.PCA(rPCAnormW,axes=c(1,3),choix="var")




# # Adresse du dossier où vous travaillez
# setwd("/net/netud/m/imoussaid/S7/Data/Data/code")

# # Packages utilisés dans la suite
# library(MASS)
# require(pls)
# require(splines)

# # Supprimer toutes les variables
# rm(list=ls(all=TRUE))

# # Utilisation de données sur data
# # Affichage des informations
# load("../Projets/superconductivity_data_train.rda")
# # Affichage des données
# print(data_train)

# # Transformation des données
# data <- data_train
# data <- data.frame(y=data$critical_temp , data )

# # Paramètres
# n <- length(data$y)
# alpha <- 0.05

# ## Mise en place de la régression linéaire [SIMPLE]
# # Peut on utiliser x1 = pourcentage de la population pauvre
# # pour prédire y = valeur médiane des maisons en milliers de dollars.
# simpleLinearReg <- lm(y~x1, data=data)

# # Affichage du résultat de la régression linéaire
# # épaisseur de la ligne = 2 ; couleur de la ligne = rouge
# plot(y~x1, data=data)
# abline(simpleLinearReg,lwd=2,col="red")

# # Affichage des résidus en fonction de la prédiction
# plot(simpleLinearReg$fitted.values, simpleLinearReg$residuals)
# abline(0,0)

# # Affichage des valeurs prédites en fonction des valeurs observées
# plot(simpleLinearReg$fitted.values,data$y)
# abline(0,1)

# # Affichage du résultat : calculer le risque à 5 % avec :
# # la t-value / la p-value/ la statistique de Fisher
# summary(simpleLinearReg)
# # Risque à 5 % (pour la t-value / la statistique de Fisher)
# qt(1-alpha/2, n-2)
# qf(1-alpha/2, 1, n-2)

# # Intervalle de confiance des paramètres estimés
# # Risque à 5 % avec l’intervalle de confiance
# confint(simpleLinearReg)

# # Adéquation au modèle avec le R^2
# summary(simpleLinearReg)

# # Prédiction d’une valeur ultérieure (valeur de x1 testée = 10)
# # Intervalle de confiance pour la prédiction de y pour une valeur donnée de x1
# predict(simpleLinearReg,data.frame(x1=10), interval="confidence")
# # Intervalle de prédiction pour la prédiction de y pour une valeur donnée de x1
# predict(simpleLinearReg,data.frame(x1=10), interval="prediction")

# # Affichage de l’intervalle de confiance et de prédiction
# seqx1 <- seq(min(data$x1),max(data$x1),length=50)
# intpred <- predict(simpleLinearReg,data.frame(x1=seqx1),
#                    interval="prediction")[,c("lwr","upr")]
# intconf <- predict(simpleLinearReg,data.frame(x1=seqx1),
#                    interval="confidence")[,c("lwr","upr")]
# plot(data$y~data$x1,xlab="x1",ylab="y")
# abline(simpleLinearReg)
# matlines(seqx1,cbind(intconf,intpred),lty=c(2,2,3,3),
#          col=c("red","red","blue","blue"),lwd=c(2,2))
# legend("bottomright",lty=c(2,3),lwd=c(2,1), c("conf","pred"),col=c("red","blue"))

# # Test de normalité des résidus
# shapiro.test(resid(simpleLinearReg))
# # Validation du modèle par validation croisée
# MSE <- 0
# for (i in 1:n)
# {
#   datatopredict <- data$y[i]
#   datatemp <- data[-c(i),]
#   reg <- lm(y~x1, data=datatemp)
#   predictedvalue <- predict(reg,data.frame(x1=data$x1[i]), interval="prediction")
#   MSE <- MSE+(datatopredict-predictedvalue[1])^2
# }
# MSE <- MSE/n
# cat("Valeur du résidu avec la validation croisée", MSE)


# ## Et le non linéaire ?
# ## Cas polynomial (simple = une variable)
# degpoly <- 7
# simplePolyReg <- lm(y~poly(x1,degpoly), data=data)
# # Risque à 5 % (pour la t-value / la statistique de Fisher)
# qt(1-alpha/2, n-2)
# qf(1-alpha/2, 1, n-2)
# # Intervalle de confiance des paramètres estimés
# # Risque à 5 % avec l’intervalle de confiance
# confint(simplePolyReg)
# # Adéquation au modèle avec le R^2
# summary(simplePolyReg)
# # Affichage de l’intervalle de confiance et de prédiction
# seqx1 <- seq(min(data$x1),max(data$x1),length=50)
# intpred <- predict(simplePolyReg,data.frame(x1=seqx1),
#                    interval="prediction")[,c("lwr","upr")]
# intconf <- predict(simplePolyReg,data.frame(x1=seqx1),
#                    interval="confidence")[,c("lwr","upr")]
# plot(data$y~data$x1,xlab="x1",ylab="y")
# pred <- predict(simplePolyReg,data.frame(x1=sort(data$x1)))
# lines(sort(data$x1),pred,lwd = 2)
# matlines(seqx1,cbind(intconf,intpred),lty=c(2,2,3,3),
#          col=c("red","red","blue","blue"),lwd=c(2,2))
# legend("bottomright",lty=c(2,3),lwd=c(2,1), c("conf","pred"),col=c("red","blue"))
# # Validation du modèle par validation croisée
# MSE <- 0
# for (i in 1:n)
# {
#   datatopredict <- data$y[i]
#   datatemp <- data[-c(i),]
#   reg <- lm(y~poly(x1,degpoly), data=datatemp)
#   predictedvalue <- predict(reg,data.frame(x1=data$x1[i]), interval="prediction")
#   MSE <- MSE+(datatopredict-predictedvalue[1])^2
# }
# MSE <- MSE/n
# # Valeur du résidu avec la validation croisée
# print(MSE)
# #
# data <- data.frame(y=data_train$critical_temp , x1=data_train$mean_atomic_mass,
#                    x2=data_train$gmean_fie)

# # Paramètres
# n <- length(data$y)
# alpha <- 0.05

# ## Mise en place de la régression linéaire [SIMPLE]
# simpleLinearReg <- lm(y~x1, data=data)

# ## Mise en place de la régression linéaire [MULTIPLE]
# linearReg <- lm(y~x1+x2, data=data)

# # Affichage du résultat
# summary(linearReg)

# # Risque à 5% : tester la nullité des coefficients du modèle de régression.
# num0fVariables <- 2
# qt(1-alpha/2, n-num0fVariables-1)
# qf(1-alpha/2, num0fVariables, n-num0fVariables-1)

# # Intervalle de confiance
# confint(linearReg)

# num0fVariablesToTest = 1
# qf(1-alpha/2, num0fVariablesToTest, n-num0fVariables-1)
# anova(simpleLinearReg,linearReg)
# simpleLinearRegx2 <- lm(y~x2, data=data)
# anova(simpleLinearRegx2,linearReg)

# # Prédiction
# predict(linearReg,data.frame(x1=10,x2=72), interval="confidence")
# predict(linearReg,data.frame(x1=10,x2=72), interval="prediction")

# # Affichage des résidus en fonction de la prédiction
# plot(linearReg$fitted.values, linearReg$residuals)
# abline(0,0)

# # Test de normalité des résidus
# shapiro.test(resid(linearReg))

# ## Selection de variables (en utilisant TOUTES les variables)
# fullLinearReg <- lm(y~., data=data)
# back <- step(fullLinearReg, direction="backward", trace = 1)
# formula(back)
# null <- lm(y ~ 1, data=data)
# forw <- step(null, scope=list(lower=null,upper=fullLinearReg),
#              direction="forward", trace = 1)
# formula(forw)

# # Utilisation de l’ACP pour réduire la dimension du problème
# # Test sur validation croisée
# # Combien de composantes pour avoir 80 % de variance expliquée ?
# redDim = pcr(y~.,data=data,scale=TRUE,validation="CV")
# summary(redDim)

# plot(data$y , redDim$comps1)
























# # Configuration de l'environnement de travail et chargement des bibliothèques
# setwd("/net/netud/m/imoussaid/S7/Data/Data/code/")
# library(class)          # Pour k-NN
# library(caret)          # Pour l'évaluation
# library(ROCR)           # Pour ROC et AUC
# library(FactoMineR)     # Pour ACP
# library(corrplot)       # Pour visualisation des corrélations
# library(MASS)           # Pour la régression
# library(pls)            # Pour régression PLS
# library(splines)        # Pour les splines dans les modèles de régression
# library("FactoMineR")
# library(PCAmixdata)
# library(GGally)


# # Suppression des variables et graphiques
# rm(list = ls(all = TRUE))
# graphics.off()

# # 1. Chargement des données d'entraînement et de test
# load("../Projets/superconductivity_data_train.rda")
# load("../Projets/superconductivity_data_test.rda")
# data_train <- na.omit(data_train)  # Suppression des valeurs manquantes
# data_test <- na.omit(data_test)

# # 2. Exploration des données
# print(head(data_train))
# summary(data_train)
# str(data_train)

# data <- data_train

# # Création des données centrées ...
# datanorm <- sweep(data,2,mean,"-")
# # # ... et réduites
# # datanorm <- sweep(datanorm,2,std,"/")
# datanorm <- sweep(data, 2, colMeans(data), "-")

# # Affichage des données centrées - réduites#  
# print(datanorm,digits=4)



# # Visualisation des données en description bivariée# 
# pairs(data[,1:10])
# # # Afficher la matrice de corrélation# 
# ggcorr(data[,1:10])
# # # Aller encore plus loin avec ggpairs# 
# ggpairs(data[,1:10])


# # # Matrice des distances entre les individus
# # dist(data_train[,1:5])
# # # Corrélation entre les variables
# # cor(data_train[,1:5])


# #Analyse en composantes principales sur les données d’origine
# # (scale.unit=FALSE)
# res <- PCA(data,graph=FALSE,scale.unit=FALSE)
# # Figure individus
# plot(res,choix="ind",cex=1.5,title="")
# # Figure variables
# plot(res,choix="var",cex=1.5,title="")


# # Analyse en composantes principales sur les données centrées-réduites
# # (par défaut: scale.unit=TRUE)
# resnorm <- PCA(data,graph=FALSE)
# # Figure individus
# plot(resnorm,choix="ind",cex=1.5,title="")
# # Figure variables
# plot(resnorm,choix="var",cex=1.5,title="")



# # # Affichage des corrélations pour les variables numériques
# # cor_matrix <- cor(data_train[, sapply(data_train, is.numeric)])
# # print(cor_matrix)
# # 
# # 
# # library(corrplot)
# # corrplot(cor_matrix, method = "circle")

# # 3. Prétraitement : Standardisation des données
# # Z scoring

# # scale_it = function(x)
# # {
# #   out = (x - mean(x)/sd(x))
# #   out
# # }

# # scaled_train = data_train
# # for(i in 1:81)
# # {
# #   scaled_train[,i] = scale_it(scaled_train[,i])
# # }




# # 4. Analyse en composantes principales (ACP)
# acp <- PCA(data_train, scale.unit = TRUE, graph = FALSE)
# summary(acp)
# plot.PCA(acp, choix = "var")  # Variables sur le plan factoriel

# # 5. Clustering K-means pour segmenter les données
# num_clusters <- 5
# kmeans_result <- kmeans(data_train_scaled, centers = num_clusters, nstart = 25)
# data_train$cluster <- kmeans_result$cluster

# # Affichage des clusters avec ACP
# data_train_pca <- PCA(data_train, quali.sup = ncol(data_train), graph = FALSE)
# plot.PCA(data_train_pca, axes = c(1, 2), choix = "ind", habillage = ncol(data_train))

# # 6. K-Nearest Neighbors (k-NN) pour la classification
# num_neighbors <- 10
# knn_pred <- knn(train = data_train_scaled, test = data_test_scaled, cl = data_train$cluster, k = num_neighbors)
# error_rate_knn <- mean(knn_pred != data_test$cluster)
# cat("Error rate using k-NN:", error_rate_knn, "\n")

# # 7. Régression Linéaire pour prédire la température critique
# # Conversion de la variable cible en un DataFrame pour la régression
# data_train$y <- data_train$critical_temp

# # Modèle de régression linéaire simple
# simple_lm <- lm(y ~ ., data = data_train)
# summary(simple_lm)

# # Visualisation des résultats de la régression
# plot(data_train$critical_temp, simple_lm$fitted.values, main = "Régression linéaire : Température critique",
#      xlab = "Température Critique Réelle", ylab = "Température Critique Prédite")
# abline(0, 1, col = "red")

# # # 8. Modèle de régression PLS (Partial Least Squares) pour réduire la dimensionnalité et faire des prédictions
# # pls_model <- plsr(y ~ ., data = data_train, scale = TRUE, validation = "CV")
# # summary(pls_model)
# # 
# # # Prédiction sur les données de test et calcul du RMSE
# # pls_pred <- predict(pls_model, data_test)
# # rmse_pls <- sqrt(mean((data_test$critical_temp - pls_pred)^2))
# # cat("RMSE for PLS Model:", rmse_pls, "\n")

# # 9. Évaluation des Modèles
# # Création de la matrice de confusion pour k-NN
# confmat_knn <- table(Predicted = knn_pred, Actual = data_test$cluster)
# print("Confusion Matrix for k-NN")
# print(confmat_knn)

# # Calcul de la sensibilité, spécificité et AUC
# TP <- confmat_knn[1, 1]
# TN <- confmat_knn[2, 2]
# FP <- confmat_knn[1, 2]
# FN <- confmat_knn[2, 1]
# TPR <- TP / (TP + FN)
# TNR <- TN / (TN + FP)
# cat("Sensitivity (TPR):", TPR, "\n")
# cat("Specificity (TNR):", TNR, "\n")

# # ROC et AUC pour k-NN
# knn_prob <- as.numeric(knn_pred)  # Conversion des prédictions
# roc_pred <- prediction(knn_prob, data_test$cluster)
# roc_perf <- performance(roc_pred, "tpr", "fpr")
# plot(roc_perf, colorize = TRUE)
# auc <- performance(roc_pred, "auc")@y.values[[1]]
# cat("AUC for k-NN:", auc, "\n")

# # 10. Visualisation des clusters avec la température critique
# plot(data_train$cluster, data_train$critical_temp, main = "Température Critique par Cluster",
#      xlab = "Clusters", ylab = "Température Critique")

# # Calcul de la moyenne et écart-type par cluster
# cluster_stats <- function(cluster_data) {
#   mean_temp <- mean(cluster_data$critical_temp)
#   sd_temp <- sd(cluster_data$critical_temp)
#   return(c(mean = mean_temp, sd = sd_temp))
# }
# clusters <- split(data_train, data_train$cluster)
# stats_list <- lapply(clusters, cluster_stats)
# stats_table <- do.call(rbind, stats_list)
# print(stats_table)











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


set.seed(123) # fixer une graine pour une reproductibilte
sample_size <- floor(nrow(scaled_train) * 20/100)
scaled_train <- scaled_train[sample(nrow(scaled_train), sample_size),]

# Modèle linéaire simple
linear_mod <- lm(critical_temp ~ . ,data = scaled_train)

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
  tmp_model = lm(critical_temp ~ . , data = tmp_train)
  tmp_y = predict(tmp_model , tmp_test[ , 1:81])
  mse <- c(mse , mean((tmp_y - tmp_test[, 82])^2))
  print(c(i , mse[i]))
}

print("full mse")
print(mean(mse))

###Deuxieme methode###
#resultat moyen que vous esayer dameliorer en metant des splines 
# resultat avec les splines a montrer aussi a la class par ex 
# resultats pas fous non pus peut faire mieux 
# comment c est fortement lineaireon peut quitter lapproche modele 
# faire par ex des foret aleatoire
# qui peuvent etre utilise pour faire aussi de la regressin

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
rf_residuals=scaled_train$critical_temp-randforest$predictions
plot(rf_residuals ~train$critical_temp, pch = 19,
     ylab = "Residuals(Obs-Pred)" , xlab="Observed critical temperatures (K)" )
abline(h=0 ,lty=2)
#3
hist(rf_residuals,   col = "grey", freq = FALSE,  xlab = "Residuals", main = "")
#c'est mieux ! maintenant c'est bcp bcp plus long...
#si on ajoute la validation croisée dessous ça devient vraiment long!
mse_rf <- c()
for (i in 1:sample_random_num)
(
 index_for_test_data=sample(1:nrpw(scaled_train),size=sample_size)
 tmp_test=scaled_train[index_for_test_data,]
 tmp_train=scaled_train[-index_for_test_data,])
 tmp_model= ranger(critical_temp ~., data=tmp_train,mtr=10,num.trees =num_trees,importance="permutation",min.node.size = 1)
 tmp=predict(randforest,tmp_test[,1:81])
 tmp_y=tmp$predictions
 mse_rf<-c(i,mse_rf[i]))
 print(c(i,mse_rf[i]))
 }
print("full mse ~ RF : ")
print(mean(mse_rf))

#quatrieme approche 
#on peut utiliser les forets que sur les premieres composante
#ACP 
n<-5 #a choisir ce nombre 
tab_train_wth_critical_temp= scaled_train[,1:81]
ACP_train<-PCA(tab_train_wth_critical_temp,graph=FALSE,ncp=ncp)
#construire les ncp nouvelles composantes dans un data frame 
ACP_dim<-as.numeric(ACP_train$var$cor[,1])
ACP_scaled_train<-data.frame(apply(scaled_train*ACP_dim,1,sum))
colnames(ACP_scaled_train)<-c(paste(c("dim",i),collapse = ""))

for (i in 2:n)
(
  ACP_dim<-as.numeric(ACP_train$var$cor[,i])
  temp_df<-scale_it(appply(scaled_train*ACP_dim,1,sum))
  ACP_scaled_train<-cbind(ACP_scaled_train,temp_df)
  colnames(ACP_scaled_train)[i]<-c(paste(c("dim",i), collapse = ""))
)

ACP_scaled_train<- cbind(ACP_scaled_train,scaled_train$critical_temp)
colnames(ACP_scaled_train)[n+1]<-c("critical_temp")

nb_trees<-100
mtry<-min(n,10)
randforest_after_ACP= ranger(critical_temp ~., data=ACP_scaled_train, mtry=mtry, min.node.size -1,
                             num.trees= nb_trees,importance="permutation")


