# massoh analyses
library(psych)
pairs.panels(data[, 5], 
gap =0, 
bg = c("yellow", "blue", "green")[data$species],
pch=21)

# pca

pc <- prcom(data,
            center=TRUE,
            scale.=TRUE)

# orthogonalyty in PCA
pairs.panels(pc$x,
             gap=0,
             bg = c("yellow", "blue", "green")[data$species],
             pch=21)

#Biplot
library(devtools)
install_github("ggbiplot", "vqv")
library(ggbiplot)
g<- ggbiplot(pc, 
             obs.scale = 1
             var.scale =1,
             groups =data$species,
             ellipse = TRUE,
             circle =TRUE,
             ellipse.prob = 0.68)
g <- g+ scale_color_discrete(names ='')
g<- g + theme(legend.direction = 'horrizontal',
              legend.position = 'top')


# another biplot

# doing it with iris data
data(iris)
head(iris)
summary(iris)
mypr<- prcomp(iris[, -5], scale =TRUE)
summary(mypr)
plot(mypr, type="l")
biplot(pc, scale =0)

str(mypr)
mypr$x
iris2<- cbind(iris, mypr$x[, 1:2])
head(iris2)

library(ggplot2)
ggplot(iris2, aes(PC1, PC2, col= species, fill =Species))
+ stat.ellipse(geom= "polygon", col = "black", alpha =0.5) +
  geom.point(shape =21, col="black")









             