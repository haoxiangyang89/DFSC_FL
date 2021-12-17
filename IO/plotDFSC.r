# plot DFSC comparison graphs, R = 24, N = 24
surgeD <- read.csv("./Desktop/Git/daniel_Diesel/output/surge24.csv", header = TRUE)
nomD <- read.csv("./Desktop/Git/daniel_Diesel/output/nom24.csv", header = TRUE)
totalD <- read.csv("./Desktop/Git/daniel_Diesel/output/total24.csv", header = TRUE)

outString = "./Desktop/Git/daniel_Diesel/output/24Plot.png"
png(file = outString, width= 8, height = 12.5, units = 'in',res = 300);
par(mfrow=c(3,1));
par(mar = c(5,5,2.5,2.5));
plot(0:384,surgeD$TotalD[1:385], type = "l", xlab = "Datetime", ylab = "Demand", main = "Surge Demand",
     col = "#000000", lwd = 2, cex.main = 1.5, cex.lab = 1.5, axes = F);
axis(cex.axis = 1.5,side=1,at=c(0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384),
    labels=c("09-06","09-07","09-08","09-09","09-10","09-11","09-12","09-13","09-14","09-15","09-16","09-17","09-18","09-19","09-20","09-21","09-22"));
axis(cex.axis = 1.5,side=2);
lines(0:384,surgeD$GEFS[1:385], col = "#377EB8", lwd = 2);
lines(0:384,surgeD$GAVG[1:385], col = "#E41A1C", lwd = 2);
lines(0:384,surgeD$NDFD[1:385], col = "#4DAF4A", lwd = 2);
lines(0:384,surgeD$REAL[1:385], col = "#984EA3", lwd = 2);
legend("topleft",c("Total","Shortage:GEFS","Shortage:GAVG","Shortage:NDFD","Shortage:PI"), , col = c("#000000","#377EB8","#E41A1C","#4DAF4A","#984EA3"),pch = 20,cex = 1.5);

par(mar = c(5,5,2.5,2.5));
plot(0:384,nomD$TotalD[1:385], type = "l", xlab = "Datetime", ylab = "Demand", main = "Nominal Demand",
     col = "#000000", lwd = 2, cex.main = 1.5, cex.lab = 1.5, axes = F);
axis(cex.axis = 1.5,side=1,at=c(0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384),
     labels=c("09-06","09-07","09-08","09-09","09-10","09-11","09-12","09-13","09-14","09-15","09-16","09-17","09-18","09-19","09-20","09-21","09-22"));
axis(cex.axis = 1.5,side=2);
lines(0:384,nomD$GEFS[1:385], col = "#377EB8", lwd = 2);
lines(0:384,nomD$GAVG[1:385], col = "#E41A1C", lwd = 2);
lines(0:384,nomD$NDFD[1:385], col = "#4DAF4A", lwd = 2);
lines(0:384,nomD$REAL[1:385], col = "#984EA3", lwd = 2);
legend("topleft",c("Total","Shortage:GEFS","Shortage:GAVG","Shortage:NDFD","Shortage:PI"), , col = c("#000000","#377EB8","#E41A1C","#4DAF4A","#984EA3"),pch = 20,cex = 1.5);

par(mar = c(5,5,2.5,2.5));
plot(0:384,totalD$TotalD[1:385], type = "l", xlab = "Datetime", ylab = "Demand", main = "Total Demand",
     col = "#000000", lwd = 2, cex.main = 1.5, cex.lab = 1.5, axes = F);
axis(cex.axis = 1.5,side=1,at=c(0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384),
     labels=c("09-06","09-07","09-08","09-09","09-10","09-11","09-12","09-13","09-14","09-15","09-16","09-17","09-18","09-19","09-20","09-21","09-22"));
axis(cex.axis = 1.5,side=2);
lines(0:384,totalD$GEFS[1:385], col = "#377EB8", lwd = 2);
lines(0:384,totalD$GAVG[1:385], col = "#E41A1C", lwd = 2);
lines(0:384,totalD$NDFD[1:385], col = "#4DAF4A", lwd = 2);
lines(0:384,totalD$REAL[1:385], col = "#984EA3", lwd = 2);
legend("topleft",c("Total","Shortage:GEFS","Shortage:GAVG","Shortage:NDFD","Shortage:PI"), , col = c("#000000","#377EB8","#E41A1C","#4DAF4A","#984EA3"),pch = 20,cex = 1.5);

dev.off()

# plot DFSC comparison graphs, R = 12, N = 12
surgeD <- read.csv("./Desktop/Git/daniel_Diesel/output/surge12.csv", header = TRUE)
nomD <- read.csv("./Desktop/Git/daniel_Diesel/output/nom12.csv", header = TRUE)
totalD <- read.csv("./Desktop/Git/daniel_Diesel/output/total12.csv", header = TRUE)

outString = "./Desktop/Git/daniel_Diesel/output/12Plot.png"
png(file = outString, width= 8, height = 12.5, units = 'in',res = 300);
par(mfrow=c(3,1));
par(mar = c(5,5,2.5,2.5));
plot(0:384,surgeD$TotalD[1:385], type = "l", xlab = "Datetime", ylab = "Demand", main = "Surge Demand",
     col = "#000000", lwd = 2, cex.main = 1.5, cex.lab = 1.5, axes = F);
axis(cex.axis = 1.5,side=1,at=c(0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384),
     labels=c("09-06","09-07","09-08","09-09","09-10","09-11","09-12","09-13","09-14","09-15","09-16","09-17","09-18","09-19","09-20","09-21","09-22"));
axis(cex.axis = 1.5,side=2);
lines(0:384,surgeD$GEFS[1:385], col = "#377EB8", lwd = 2);
lines(0:384,surgeD$GAVG[1:385], col = "#E41A1C", lwd = 2);
lines(0:384,surgeD$NDFD[1:385], col = "#4DAF4A", lwd = 2);
lines(0:384,surgeD$REAL[1:385], col = "#984EA3", lwd = 2);
legend("topleft",c("Total","Shortage:GEFS","Shortage:GAVG","Shortage:NDFD","Shortage:PI"), , col = c("#000000","#377EB8","#E41A1C","#4DAF4A","#984EA3"),pch = 20,cex = 1.5);

par(mar = c(5,5,2.5,2.5));
plot(0:384,nomD$TotalD[1:385], type = "l", xlab = "Datetime", ylab = "Demand", main = "Nominal Demand",
     col = "#000000", lwd = 2, cex.main = 1.5, cex.lab = 1.5, axes = F);
axis(cex.axis = 1.5,side=1,at=c(0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384),
     labels=c("09-06","09-07","09-08","09-09","09-10","09-11","09-12","09-13","09-14","09-15","09-16","09-17","09-18","09-19","09-20","09-21","09-22"));
axis(cex.axis = 1.5,side=2);
lines(0:384,nomD$GEFS[1:385], col = "#377EB8", lwd = 2);
lines(0:384,nomD$GAVG[1:385], col = "#E41A1C", lwd = 2);
lines(0:384,nomD$NDFD[1:385], col = "#4DAF4A", lwd = 2);
lines(0:384,nomD$REAL[1:385], col = "#984EA3", lwd = 2);
legend("topleft",c("Total","Shortage:GEFS","Shortage:GAVG","Shortage:NDFD","Shortage:PI"), , col = c("#000000","#377EB8","#E41A1C","#4DAF4A","#984EA3"),pch = 20,cex = 1.5);

par(mar = c(5,5,2.5,2.5));
plot(0:384,totalD$TotalD[1:385], type = "l", xlab = "Datetime", ylab = "Demand", main = "Total Demand",
     col = "#000000", lwd = 2, cex.main = 1.5, cex.lab = 1.5, axes = F);
axis(cex.axis = 1.5,side=1,at=c(0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384),
     labels=c("09-06","09-07","09-08","09-09","09-10","09-11","09-12","09-13","09-14","09-15","09-16","09-17","09-18","09-19","09-20","09-21","09-22"));
axis(cex.axis = 1.5,side=2);
lines(0:384,totalD$GEFS[1:385], col = "#377EB8", lwd = 2);
lines(0:384,totalD$GAVG[1:385], col = "#E41A1C", lwd = 2);
lines(0:384,totalD$NDFD[1:385], col = "#4DAF4A", lwd = 2);
lines(0:384,totalD$REAL[1:385], col = "#984EA3", lwd = 2);
legend("topleft",c("Total","Shortage:GEFS","Shortage:GAVG","Shortage:NDFD","Shortage:PI"), , col = c("#000000","#377EB8","#E41A1C","#4DAF4A","#984EA3"),pch = 20,cex = 1.5);

dev.off()