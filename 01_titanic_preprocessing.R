library(ProjectTemplate)
library(here)

create.project(project.name="titanic",
               template="minimal")


setwd("/Users/keithpost/Documents/R files/Kaggle Competitions/titanic")
getwd()
here()

library("tidyverse")
library("tree")
library("caret")

#Some ideas to consider
#1) grouping Names into families and adding family as another variable 
#2) see if family size correlates in some way to the cabin letter code
#3) see if ticket numbers (e.g., starting with C.A., PC, SOTON) are associated in some way to cabin letter code, fare, survival, or another variable
#4) see if passengers with significant titles (e.g., doctor, colonel, madame) are associated with particular cabins, tickets, pclass, etc.





#I. Data import
#**************
titanic<-read_csv("train.csv")
glimpse(titanic)
summary(titanic)
head(titanic,n=20)


#Parse vectors using parse_* (for character vectors)
titanic$Sex<-parse_factor(titanic$Sex)
titanic$Embarked<-parse_factor(titanic$Embarked)


#Other parsings needed
titanic$PassengerId<-as.character(titanic$PassengerId) 
#converts PassengerId to character type b/c it's a unique identifier
titanic$Survived<-as.logical(titanic$Survived)
#converts Survived to logical because data are binary
titanic$Pclass<-factor(titanic$Pclass,exclude=NULL) 
#converts Pclass from dbl to factor because there are three SES categories
titanic$SibSp<-as.integer(titanic$SibSp)
#converts SibSp from dbl to integer because SibSp represent counts
titanic$Parch<-as.integer(titanic$Parch)
#converts Parch from dbl to integer because Parch represent counts

#partition data to training and probe datasets
set.seed(33)
inTrain<-createDataPartition(y=titanic$Survived,p=0.75,list=FALSE)
titanic<-titanic[inTrain,]
probe<-titanic[-inTrain,]



#II. Tidy data
#*************
glimpse(titanic) #variables are in the correct format, and data are organized correctly (each variable has a column, each observation has a row, and each value has a cell); thus, no pivoting needed; no separating or uniting necessary either
#thus, data are tidy (even though there are NAs)
#also note that the name column could be separated out for further variables--such as special titles, like madame, colonel, doctor, etc.

#Create new variable (family size)
#titanic<-mutate(titanic,Family=as.numeric(SibSp)+as.numeric(Parch))
#titanic$Family<-factor(titanic$Family)


#III. Data transformation and Cleaning
#*************************************
#Count NAs
summary(titanic) #look for NAs: Age (140), Embarked (2)
filter(titanic,is.na(PassengerId)) #0
filter(titanic,is.na(Name)) #0
filter(titanic,is.na(Ticket)) #0
filter(titanic,is.na(Cabin)) #530 NAs
titanic<-titanic[,-11] #removes Cabin

#Determine SDs
sd(titanic$Age,na.rm=TRUE) #14.48 (mean=29.84)
sd(titanic$SibSp) #1.15 (mean=0.5291)
hist(titanic$SibSp)
sd(titanic$Parch) #.806 (mean=0.38)
hist(titanic$Parch)
hist(titanic$Fare)

#Transformation: normalization of SibSp and Parch
preObj<-preProcess(titanic[,-2],method=c("center","scale"))
titanic$SibSp<-predict(preObj,titanic[,-2])$SibSp
titanic$Parch<-predict(preObj,titanic[,-2])$Parch
titanic$Fare<-predict(preObj,titanic[,-2])$Fare

mean(titanic$SibSp); sd(titanic$SibSp)
mean(titanic$Parch); sd(titanic$Parch)
mean(titanic$Fare); sd(titanic$Fare)

#Data Imputation
preObj2<-preProcess(titanic[,-2],method="knnImpute") 
titanic$Age<-predict(preObj2,titanic)$Age 

summary(titanic) #notice that Age no longer has NAs but now data are standardized

#need a strategy to impute Embarked NAs


#Select (reorder) and arrange variables
titanic<-select(titanic,PassengerId,Name,Sex,Age,Pclass,Embarked,Ticket,Fare,SibSp,Parch,Survived)
titanic<-arrange(titanic,Name)


#IV. Visualization/Exploratory Data Analysis
#*******************************************
#Visualize distributions (and look for common and unusual values)
#categorical
ggplot(data=titanic)+geom_bar(aes(Sex)) #almost 2x males than females
ggplot(data=titanic)+geom_bar(aes(Pclass)) #almost 2.5x class 3 than 1 or 2
ggplot(data=titanic)+geom_bar(aes(Embarked)) #~2x C than Q and ~4x S than C (note that there are some NAs)

cabin.data<-na.omit(titanic$Cabin) #removes NAs from Cabin
sum(str_detect(cabin.data,"A")) #10
sum(str_detect(cabin.data,"B")) #35
sum(str_detect(cabin.data,"C")) #34
sum(str_detect(cabin.data,"D")) #21
sum(str_detect(cabin.data,"E")) #26
sum(str_detect(cabin.data,"F")) #10
sum(str_detect(cabin.data,"G")) #5
cabin.counts<-c(10,35,34,21,26,10,5)
cabin.letters<-c("A","B","C","D","E","F","G")
cabin.means<-data.frame(cabin.letters,cabin.counts)
ggplot(cabin.means)+geom_bar(aes(x=cabin.letters,y=cabin.counts),stat="identity")

ggplot(data=titanic)+geom_bar(aes(SibSp)) #~600 0, ~150 1, and 2-5 & 8 <50
ggplot(data=titanic)+geom_bar(aes(Parch)) #~525 0, ~75 1, ~75 2, & 3-6 near-0
#most people travelled alone
(titanic.alone<-filter(titanic,SibSp==0,Parch==0)) #407 meet these criteria
titanic.alone %>% count(Sex,Survived) #over 5x males died than lived, and nearly 4x females lived than died
summarize(titanic.alone,mean(Age,na.rm=TRUE)); summarize(titanic,mean(Age,na.rm=TRUE)) #solo passengers slightly older on average (32.2 yo vs 29.7 yo...maybe this is due to a different m:f)

ggplot(data=titanic)+geom_bar(aes(Survived)) #~425 Died & ~250 Survived

#continuous
ggplot(data=titanic)+geom_histogram(aes(Age),binwidth=2) #right-skewed normal with a valley ~10 yo and higher than expected <1 yo
ggplot(data=titanic)+geom_histogram(aes(Fare)) #fares generally under $275 except for at least one >$500; resembles Poisson distribution


#Covariation: cat-cont, cat-cat, cont-cont
#categorical-continuous
#Sex vs. Age
ggplot(data=titanic,aes(Age))+geom_freqpoly(aes(color=Sex)) #tough to compare patterns because of differences in counts between ages
#let's use density on y-axis to make comparison easier
ggplot(data=titanic,aes(Age,..density..))+geom_freqpoly(aes(color=Sex)) #no difference between sexes

#do the same for other categorical data
#Pclass vs. Age
ggplot(data=titanic,aes(Age))+geom_freqpoly(aes(color=Pclass))
ggplot(data=titanic,aes(Age,..density..))+geom_freqpoly(aes(color=Pclass))
#Pclasses 2 and 3 have small peak at young ages; Pclass 1 has max at older age than 2 and 3

#Embarked vs. Age
ggplot(data=titanic,aes(Age))+geom_freqpoly(aes(color=Embarked))
ggplot(data=titanic,aes(Age,..density..))+geom_freqpoly(aes(color=Embarked)) #shows NAs at just under 40 and just over 60 yo
ggplot(data=titanic)+geom_freqpoly(aes(Age,..density..,color=Embarked),na.rm=TRUE)
#trying to remove NAs for this plot but can't get it to work...not a huge difference in distribution of ages among Embarked categories

#SibSp vs. Age
ggplot(data=titanic,aes(Age))+geom_freqpoly(aes(color=SibSp))
ggplot(data=titanic,aes(Age,..density..))+geom_freqpoly(aes(color=SibSp))
#when adjusted for density...divergent patterns among SibSp cats
ggplot(data=titanic,aes(Age,..density..))+geom_freqpoly()+facet_wrap(~ SibSp) #now beter able to see divergent patterns...which is that ages condense at lower levels as SibSp increases

#Parch vs. Age
ggplot(data=titanic,aes(Age))+geom_freqpoly(aes(color=Parch))
ggplot(data=titanic,aes(Age,..density..))+geom_freqpoly(aes(color=Parch))
#5 and 6 have maxes around 40
ggplot(data=titanic,aes(Age,..density..))+geom_freqpoly()+facet_wrap(~ Parch) #ages condense at around 40 as Parch increases

#Sex vs. Fare
ggplot(data=titanic,aes(Fare))+geom_freqpoly(aes(color=Sex))
ggplot(data=titanic,aes(Fare,..density..))+geom_freqpoly(aes(color=Sex)) #very similar patterns...but this includes those at least one high fare
titanic %>% filter(Fare<500) %>%
	ggplot(aes(Fare,..density..))+geom_freqpoly(aes(color=Sex))
#after excluding high fares, patterns are similar between sexes

#Pclass vs. Fare
ggplot(data=titanic,aes(Fare))+geom_freqpoly(aes(color=Pclass))
ggplot(data=titanic,aes(Fare,..density..))+geom_freqpoly(aes(color=Pclass)) #all three classes have long right tails; maxes are (in increasing order) Pclass 3, then 2, then 1

#Embarked vs. Fare
ggplot(data=titanic,aes(Fare))+geom_freqpoly(aes(color=Embarked))
ggplot(data=titanic,aes(Fare,..density..))+geom_freqpoly(aes(color=Embarked))
#all three have maxes at low fares

#SibSp vs. Fare
ggplot(data=titanic,aes(Fare))+geom_freqpoly(aes(color=SibSp))
ggplot(data=titanic,aes(Fare,..density..))+geom_freqpoly(aes(color=SibSp))
ggplot(data=titanic,aes(Fare,..density..))+geom_freqpoly()+facet_wrap(~ SibSp) #faceting to illustrate patterns better...appears that there are similar patterns for low SibSp and that high SibSp have are rarer
by_SibSp<-group_by(titanic,SibSp)
summarize(by_SibSp,count=n()) #these counts show that only 64 passengers had 2 or greater SibSp

#Parch vs. Fare
ggplot(data=titanic,aes(Fare))+geom_freqpoly(aes(color=Parch))
ggplot(data=titanic,aes(Fare,..density..))+geom_freqpoly(aes(color=Parch))
ggplot(data=titanic,aes(Fare,..density..))+geom_freqpoly()+facet_wrap(~ Parch) #similar pattern among all Parch levels (peaks at very low fare) except that Parch=2 indicates a smaller peak and more observations at higher fares


#categorical-categorical (e.g., Sex, Pclass, Embarked, SibSp, Parch)
#visualize covariation by plotting the number of observations for each combination of variables
#Sex vs. other variables
ggplot(titanic)+geom_count(aes(Sex,Pclass)) #most = m-3; least = f-2; more m than f for each Pclass
ggplot(titanic)+geom_count(aes(Sex,Embarked)) #Q and C similar b/t sexes; more m than f for S
ggplot(titanic)+geom_count(aes(Sex,SibSp)) #more m than f for 0; otherwise similar b/t sexes for each SibSp
ggplot(titanic)+geom_count(aes(Sex,Parch)) #more m than f for 0; more f than f for 6; otherwise similar b/t sexes for each Parch

#Pclass vs. other variables
ggplot(titanic)+geom_count(aes(Pclass,Embarked)) #more S than C/Q; more 3 than 1/2
ggplot(titanic)+geom_count(aes(Pclass,SibSp)) #more 3 (Pclass) than 1/2 for 4, 5, or 8 SibSp; more 0 SibSp than 1-5 or 8 SibSp (per level)
ggplot(titanic)+geom_count(aes(Pclass,Parch)) #similar to SibSp 

#Embarked vs. other variables
ggplot(titanic)+geom_count(aes(Embarked,SibSp)) #observations tend to increase as one goes from top-right (8-Q) to lower left (0-S)
ggplot(titanic)+geom_count(aes(Embarked,Parch)) #similar to above

#SibSp vs. Parch
ggplot(titanic)+geom_count(aes(SibSp,Parch)) #observations tend to increase from large-large to small-small


#continuous-continuous
ggplot(data=titanic)+geom_point(aes(Age,Fare)) #fare outliers (>$500) for two people just shy of 40 yo; no obvious relationship here
ggplot(data=titanic)+geom_point(aes(Age,Fare,color=Pclass)) #lower class passengers generally had lower ticket fares per age, and fares increased with each class at the same age level
ggplot(data=titanic)+geom_point(aes(Age,Fare,color=Sex))+geom_smooth(aes(Age,Fare,color=Sex)) #for a given age, females have a higher fare (and males had a larger age range)


#covariation with dependent variable (Survived)
#categorical-continuous
ggplot(data=titanic,aes(Age))+geom_freqpoly(aes(color=Survived))
ggplot(data=titanic,aes(Age,..density..))+geom_freqpoly(aes(color=Survived))
#similar patterns (normalish)...except that survivors had a small peak at very young ages when non-survivors didn't and non-survivors had a greater max density than survivors (at ~20 yo)

ggplot(data=titanic,aes(Fare))+geom_freqpoly(aes(color=Survived))
ggplot(data=titanic,aes(Fare,..density..))+geom_freqpoly(aes(color=Survived)) #very similar pattern (Poisson)...except that non-survivors had a greater max density (around 0 yo) and generally smaller densities for the remainder of the curve

#categorical-categorical
ggplot(titanic)+geom_count(aes(Survived,Sex)) #more females survived than did not survive, while the opposite pattern in males
ggplot(titanic)+geom_count(aes(Survived,Pclass)) #as one goes from 1->3 Pclass, survivors>non-survivors then similar then opposite
ggplot(titanic)+geom_count(aes(Survived,Embarked)) #for S, more non-s than s; other Embarked groups similar b/t survivor groups
ggplot(titanic)+geom_count(aes(Survived,SibSp)) #similar b/t survivor groups for 0-4; more non-s than s for 5-6
ggplot(titanic)+geom_count(aes(Survived,Parch)) #similar pattern to SibSp


#Other analyses
summary(lm(as.numeric(Parch)~as.numeric(SibSp),titanic)) #significant...but R^2 =0.18 (no strong evidence of collinearity)

#**********working on this********
#4) see if passengers with significant titles (e.g., doctor, colonel, madame) are associated with particular cabins, tickets, pclass, etc.
#master, reverend (Rev.), Capt., Lady., Sir., Dr.,  


#V. Modelling
#simple logistic models
summary(glm(titanic$Survived~titanic$Sex,family=binomial("logit"))) #significant p; AIC = 708
titanic.SexSurv<-glm(titanic$Survived~titanic$Sex,binomial("logit"))
anova(titanic.SexSurv,test="Chisq")

#regression tree
titanic.model<-tree(Survived~.,titanic)
print(titanic.model)


#VI. Process Probe Data
glimpse(probe)
summary(probe)

#Count NAs
summary(probe) #look for NAs: Age (27), Embarked (1)
filter(probe,is.na(PassengerId)) #0
filter(probe,is.na(Name)) #0
filter(probe,is.na(Ticket)) #0
filter(probe,is.na(Cabin)) #127 NAs
probe<-probe[,-11] #removes Cabin

#Determine SDs
sd(probe$Age,na.rm=TRUE) #13.82 (mean=29.38)
sd(probe$SibSp) #0.969 (mean=0.586)
hist(probe$SibSp)
sd(probe$Parch) #.744 (mean=0.358)
hist(probe$Parch)
hist(probe$Fare)

#********Start Up Here*****
#Transformation: normalization of SibSp and Parch
preObj<-preProcess(titanic[,-2],method=c("center","scale"))
probe$SibSp<-predict(preObj,probe[,-2])$SibSp
probe$Parch<-predict(preObj,probe[,-2])$Parch
probe$Fare<-predict(preObj,probe[,-2])$Fare

mean(probe$SibSp); sd(probe$SibSp)
mean(probe$Parch); sd(probe$Parch)
mean(probe$Fare); sd(probe$Fare)

#Data Imputation
preObj2<-preProcess(titanic[,-2],method="knnImpute") 
probe$Age<-predict(preObj2, probe)$Age 

summary(probe) #notice that Age no longer has NAs but now data are standardized

#need a strategy to impute Embarked NAs


#Select (reorder) and arrange variables
probe<-select(probe,PassengerId,Name,Sex,Age,Pclass,Embarked,Ticket,Fare,SibSp,Parch,Survived)
probe<-arrange(probe,Name)


#VII; Test Model on Probe Data









