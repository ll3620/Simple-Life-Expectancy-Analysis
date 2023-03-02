library(imputeTS)
library('psych')
library(ggplot2)
library(dplyr)
library(tidyr)
library(reshape2)
library(caTools)
library(caret)
library("pls")
library(tidyverse)
library(glmnet)
library(ggfortify)
library(glmnetUtils)
library(plot3D)


###
setwd("E:/Imperial/Year 3 Modules/MATH60049-IntroToStatsLearning")
full.data <- read.csv("Life_Expectancy_Data.csv")
Reduced.data <- full.data[full.data$Year %in% c(2015),]
Reduced.data <- subset(Reduced.data, select = -c(Status, Year, Alcohol, percentage.expenditure, Total.expenditure, under.five.deaths))

rownames(Reduced.data) <- Reduced.data$Country
Reduced.data <- Reduced.data[, -which(names(Reduced.data) == 'Country')]
Clean.data <- na.omit(Reduced.data)

# Select all columns except sunshine
cols_to_standardize <- Clean.data[, !colnames(Clean.data) %in% "Life.expectancy"]
# Standardize columns
Clean.data[, !colnames(Clean.data) %in% "Life.expectancy"] <- scale(cols_to_standardize)

##### Pre-Analysis Visualization
corr_matrix <- cor(Clean.data)
corr_long <- melt(corr_matrix)
# Plot heatmap
ggplot(corr_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(name = "Corr", low = "dodgerblue4", mid = "white", high = "firebrick4",
                       midpoint = 0, limits = c(-1, 1), guide = "colorbar") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  labs(title = "Correlation between features") +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab(NULL) +
  ylab(NULL)

##### Simple Linear Regression to check assumptions
eq.lm <- lm(Life.expectancy~., data=Clean.data)
summary(eq.lm)

eq.lm2 <- update(eq.lm, .~. -infant.deaths-Measles-BMI-Polio-Diphtheria-GDP-Population-thinness..1.19.years-thinness.5.9.years-Schooling)
summary(eq.lm2)

plot(eq.lm2)
plot(eq.lm2, which=4)

########################
x <- model.matrix(Life.expectancy~., Clean.data)[,-1]
y <- Clean.data$Life.expectancy
lambda <- 10^seq(from=10, to=-2, length=100)

set.seed(88)
train <- sample(1:nrow(x), nrow(x)/2)
test <- -train
ytest <- y[test]

# Simple Linear Regression
data.train.lm <- lm(Life.expectancy~., data=Clean.data, subset=train)
summary(data.train.lm)

data.train.lm <- update(data.train.lm, .~. -infant.deaths-Measles-BMI-HIV.AIDS-GDP-thinness..1.19.years-thinness.5.9.years-Schooling)
summary(data.train.lm)

plot(data.train.lm)

# Ridge
data.train.ridge <- glmnet(x[train,], y[train], alpha=0, lambda=lambda)
cv.data.ridge <- cv.glmnet(x[train,], y[train], alpha=0)
bestlam.ridge <- cv.data.ridge$lambda.min

# Lasso
data.train.lasso <- glmnet(x[train,], y[train], alpja=1, lambda=lambda)
cv.lasso <- cv.glmnet(x[train,], y[train], alpha=1)
bestlam.lasso <- cv.lasso$lambda.min

# PCR
prcomp(x)
cumsum(prcomp(x)$sdev^2) / sum(prcomp(x)$sdev^2)
biplot(prcomp(Clean.data), scale=-1, col=c('blue','red'), arrow.len=0.1)

data.train.pcr <- pcr(Life.expectancy~., data=Clean.data, validation="CV")
validationplot(data.train.pcr)
mtext(paste0("Variances Explained: ", toString(round(cumsum(prcomp(x)$sdev^2) / sum(prcomp(x)$sdev^2), 2))), side = 3, line = 0.5)

#################################################################################################################

## Prediction and evaluation

mse <- function(y1, y2) {mean((y1-y2)^2)}

data.predict.lm <- predict(data.train.lm, newdata=Clean.data[test,])

mse.lm <- mse(ytest, data.predict.lm)

data.predict.ridge <- predict(data.train.ridge, s=bestlam.ridge, newx=x[test,])
mse.ridge <- mse(ytest, data.predict.ridge)

data.predict.lasso <- predict(data.train.lasso, s=bestlam.lasso, newx=x[test,])
mse.lasso <- mse(ytest, data.predict.lasso)

data.predict.pcr <- predict(data.train.pcr, newdata = Clean.data[test,], ncomp=9)
mse.pcr <- mse(ytest, data.predict.pcr)


#################################################################################################################

## Coefficients

# Linear Regression
coef.lm <- as.matrix(coef(data.train.lm))

# Ridge
coef.ridge <- as.matrix(predict(data.train.ridge, type='coefficients', s=bestlam.ridge))

# Lasso
coef.lasso <- as.matrix(predict(data.train.lasso, type='coefficients', s=bestlam.lasso))


# Table of coefficients
coef_mat <- cbind(Parameter = rownames(coef.lm), coef.lm, coef.ridge, coef.lasso)
colnames(coef_mat)[2:4] <- c("Linear Regression", "Ridge Regression", "Lasso Regression")
coef_mat

#################################################################################################################

# Plot residuals
lm.res <- ytest - data.predict.lm
#ri.res <- ytest - data.predict.ridge
la.res <- ytest - data.predict.lasso
pcr.res <- ytest - data.predict.pcr
ylim <- range(c(lm.res, la.res, pcr.res))
plot(ytest, lm.res, ylim = ylim, ylab = "Residuals", xlab = "True Y Test Values")
#points(ytest, ri.res, col = 2)
points(ytest, la.res, col = 3)
points(ytest, pcr.res, col=4)
abline(h = 0, lty = 2)
n <- length(pcr.res)
ytest <- ytest
for (i in 1:n) {
  rvec <- c(lm.res[i], la.res[i], pcr.res[i])
  ix <- which(abs(rvec) == min(abs(rvec)))
  lines(c(ytest[i], ytest[i]), c(0, rvec[ix]), col = ix, lty = 2)
}
legend(x = "bottomright", col = 1:4, legend = c("Least Squares", "Lasso", "PCR"), pch = 0.5, cex = 0.8)

#################################################################################################################
data.predict.check <- predict(data.train.lasso, newdata=Clean.data[train,])
# Residual plot
lasso_resid_df <- data.frame(Actual = y[train], Predicted = data.predict.check)

# Add residuals column
lasso_resid_df$Residuals <- lasso_resid_df$Actual - lasso_resid_df$Predicted

# Create residual plot
ggplot(lasso_resid_df, aes(x = Predicted, y = Residuals)) +
  geom_point(color = "#1F77B4", alpha = 0.8, size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#D62728", size = 1) +
  geom_smooth(method = "loess", se = FALSE, color = "#FF7F0E", size = 1.2) +
  labs(title = "Residuals vs Fitted Plot for Lasso Model",
       x = "Fitted Values",
       y = "Residuals") +
  theme_minimal() +
  theme(plot.title = element_text(size = 18, face = "bold"),
        plot.subtitle = element_text(size = 14),
        plot.caption = element_text(size = 10),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        legend.position = "none")


# Obtain residuals from lasso model
lasso_resid <- y[train] - data.predict.check

# Calculate mean and standard deviation of residuals
resid_mean <- mean(lasso_resid)
resid_sd <- sd(lasso_resid)

# Standardize residuals
lasso_std_resid <- lasso_resid / resid_sd

qq_data <- data.frame(Theoretical_Quantiles = qnorm((1:length(lasso_std_resid)) / (length(lasso_std_resid) + 1)),
                      Sample_Quantiles = sort(lasso_std_resid))

# Create QQ plot using ggplot
ggplot(qq_data, aes(x = Theoretical_Quantiles, y = Sample_Quantiles)) +
  geom_point(color = "#1F77B4", alpha = 0.8, size = 3) +
  geom_abline(intercept = 0, slope = 1, color = "#FF7F0E", size = 1) +
  labs(title = "QQ Plot of Standardized Residuals for Lasso Model",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles") +
  theme_minimal() +
  theme(plot.title = element_text(size = 18, face = "bold"),
        plot.subtitle = element_text(size = 14),
        plot.caption = element_text(size = 10),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        legend.position = "none")

# Create data frame for scale-location plot
lasso_scale_df <- data.frame(Fitted = data.predict.check, StdResid = lasso_std_resid)

# Create scale-location plot
ggplot(lasso_scale_df, aes(x = Fitted, y = sqrt(abs(StdResid)))) +
  geom_point(color = "#1F77B4", alpha = 0.8, size = 3) +
  geom_smooth(method = "loess", se = FALSE, color = "#FF7F0E", size = 1.2) +
  labs(title = "Scale-Location Plot for Lasso Model",
       x = "Fitted Values",
       y = expression(sqrt("|Standardized Residuals|")))+
  theme_minimal() +
  theme(plot.title = element_text(size = 18, face = "bold"),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14))


#################################################
# Create a data frame for the predicted values
pred_df <- data.frame(x = data.predict.check, y = y[train])

# Create a scatter plot of the predicted vs. actual values
ggplot(pred_df, aes(x = x, y = y)) +
  geom_point(color = "#1F77B4", alpha = 0.8, size = 3) +
  geom_smooth(method = "lm", se = TRUE, color = "#FF7F0E", size = 1.2) +
  labs(title = "Data Fitting of Lasso Model",
       x = "Predicted Values",
       y = "Actual Values") +
  theme_minimal() +
  theme(plot.title = element_text(size = 18, face = "bold"),
        plot.subtitle = element_text(size = 14),
        plot.caption = element_text(size = 10),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        legend.position = "none")

#######################
resid <- lasso_std_resid
# Obtain leverage and Cook's distance
leverage <- hatvalues(data.train.lm)
cooksd <- cooks.distance(data.train.lm)

# Create data frame for leverage, standardized residuals, and Cook's distance
lasso_resid_lev_df <- data.frame(StdResid = lasso_std_resid, Leverage = leverage)

# Add Cook's distance column
lasso_resid_lev_df$Cooksd <- cooksd

# Plot standardized residuals vs leverage with Cook's distance values of 0.5 and 1 highlighted
plot(resid ~ leverage, main = "Standardized Residuals vs Leverage",
     xlab = "Leverage", ylab = "Standardized Residuals")
abline(h = c(0, -2, 2), col = "gray", lty = 2)
points(resid ~ leverage, pch = 20, cex = 1.5, col = ifelse(cooksd > 0.5, "red", "black"))
points(resid[which.max(cooksd)] ~ leverage[which.max(cooksd)], pch = 20, cex = 2, col = "orange")
legend("topright", legend = c("Cook's Distance > 0.5", "Max Cook's Distance"), pch = 20,
       col = c("red", "orange"), cex = 0.8)


# Plot standardized residuals vs leverage with Cook's distance values of 0.5 and 1 highlighted
lasso_resid_df <- lasso_resid_df[!row.names(lasso_resid_df) %in% "Trinidad and Tobago",]
ggplot(data = lasso_resid_lev_df, aes(x = Leverage, y = StdResid)) +
  geom_point(color = "#1F77B4", alpha = 0.8, size = 3) +
  geom_text(aes(label = ifelse(Cooksd > 1, rownames(lasso_resid_lev_df), "")),
            hjust = 0, vjust = 0, size = 3, col = "orange") +
  geom_text(aes(label = ifelse(Cooksd > 1, rownames(lasso_resid_lev_df), "")),
            hjust = 0, vjust = 0, size = 3, col = "orange") +
  geom_smooth(method = "loess", se = FALSE, col = "#FF7F0E") +
  scale_color_identity(guide = "legend", labels = c("Cook's Distance > 0.5"), 
                       breaks = "red") +
  labs(x = "Leverage", y = "Standardized Residuals",
       title = "Standardized Residuals vs Leverage",
       subtitle = "Highlighting Cook's Distance > 0.5",
       caption = "Source: Lasso Regression Model") +
  theme_minimal() +
  theme(plot.title = element_text(size = 18, face = "bold"),
        plot.subtitle = element_text(size = 14),
        plot.caption = element_text(size = 10),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        legend.position = "none")

###########################
#3D plots
windows(width = 10, height = 8)

# Scatterplot matrix for Clean.data
scatter3D(x = Clean.data$Adult.Mortality, y = Clean.data$Income.composition.of.resources, z = Clean.data$Life.expectancy,
          colvar = as.numeric(as.factor(row.names(Clean.data))),
          bty='g',
          colkey=FALSE,
          xlab = "Adult Mortality", ylab = "Income Composition of Resources", zlab = "Life Expectancy",
          main = "3D Scatterplot of Life Expectancy")
# Add labels for top and bottom three countries
text3D(Clean.data$Adult.Mortality, Clean.data$Income.composition.of.resources, Clean.data$Life.expectancy,  labels = rownames(Clean.data),
       add = TRUE, colkey = FALSE, cex = 0.5)

