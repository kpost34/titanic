#load packages
library(readr)
library(here)
library(skimr)
library(dplyr)
library(ggplot2)
library(cowplot)
library(PerformanceAnalytics)
library(GGally)
library(knitr)
library(kableExtra)



#III. Data exploration/visualization
#**********************************
#import cleaned and tidied data as a .csv to a tibble
f_titanic<-read_csv(here("data","tidy_data","tidy_train.csv"),
         col_types="fifniinfffff")

#Summary stats & graphs
#summary statistics
summary(f_titanic)
skim(f_titanic) 

#correlation matrix
#correlation plots of integer/numeric vars
cor(f_titanic[,4:7])
f_titanic %>%
  select(age:fare) %>%
  chart.Correlation()
f_titanic %>%
  select(age:fare) %>%
  ggcorr(label=TRUE)
f_titanic %>%
  select(age:fare) %>%
  ggpairs()

#Predictors only
#Univariate
#categorical
ggplot(data=f_titanic)+geom_bar(aes(survived)) #~425 Died & ~250 Survived
ggplot(data=f_titanic)+geom_bar(aes(sex_marital_status)) #almost 2x males than females
ggplot(data=f_titanic)+geom_bar(aes(pclass)) #almost 2.5x class 3 than 1 or 2
ggplot(data=f_titanic)+geom_bar(aes(embarked)) #~2x C than Q and ~4x S than C (note that there are some NAs)

g1<-ggplot(data=f_titanic)+geom_bar(aes(sib_sp)) #~600 0, ~150 1, and 2-5 & 8 <50
g2<-ggplot(data=f_titanic)+geom_bar(aes(parch)) #~525 0, ~75 1, ~75 2, & 3-6 near-0
plot_grid(g1,g2,ncol=1)
#most people travelled alone

#continuous
p1<-ggplot(data=f_titanic)+geom_histogram(aes(age),binwidth=2) #right-skewed normal with a valley ~10 yo and higher than expected <1 yo
p2<-ggplot(data=f_titanic)+geom_density(aes(age))
p3<-ggplot(data=f_titanic)+geom_histogram(aes(fare)) #fares generally under $275 except for at least one >$500; resembles Poisson distribution
p4<-ggplot(data=f_titanic)+geom_density(aes(fare))
plot_grid(p1,p2,p3,p4,ncol=2)
#age is close to normal except more newborns/toddlers and fewer teens than expected 
#perhaps fare could be modeled with a log-normal or gamma distrubition; notice a couple fares > $500


#Bivariate
#categorical-categorical
ggplot(f_titanic)+geom_count(aes(sib_sp,parch)) #sib_sp vs. parch; observations tend to increase from large-large to small-small
ggplot(f_titanic)+geom_count(aes(pclass,parch)) #pclass-parch

#categorical-continuous
ggplot(data=f_titanic) +
  geom_boxplot(aes(pclass,age))
#age by pclass: age tends to decrease as pclass increases

#categorical/integer-continuous
ggplot(data=f_titanic)+geom_point(aes(age,fare)) #fare outliers (>$500) for two people just shy of 40 yo; no obvious relationship here
ggplot(data=f_titanic)+geom_point(aes(age,fare,color=pclass)) #lower class passengers generally had lower ticket fares per age, and fares increased with each class at the same age level
ggplot(data=f_titanic,aes(fare,..density..))+geom_freqpoly()+facet_wrap(~ parch) #similar pattern among all Parch levels (peaks at very low fare) except that Parch=2 indicates a smaller peak and more observations at higher fares
ggplot(data=f_titanic,aes(age,..density..))+geom_freqpoly()+facet_wrap(~ sib_sp) #now beter able to see divergent patterns...which is that ages condense at lower levels as SibSp increases
ggplot(data=i_titanic,aes(age,..density..))+geom_freqpoly()+facet_wrap(~ parch) #ages condense at around 40 as Parch increases
ggplot(data=f_titanic,aes(fare,..density..))+geom_freqpoly(aes(color=embarked))
ggplot(data=f_titanic,aes(fare,..density..))+geom_freqpoly()+facet_wrap(~ sib_sp)

#continuous-continuous
ggplot(f_titanic) + geom_point(aes(age,fare))+geom_smooth(aes(age,fare),method="lm")


#Predictors and dependent variable (survived)
#categorical-continuous
ggplot(data=f_titanic) + 
  geom_point(aes(sib_sp,fare,color=survived)) +
  facet_wrap(~survived)
#differences seem to occur from 3+

s1<-ggplot(data=f_titanic,aes(age,..density..)) + 
  geom_freqpoly(aes(color=survived)) +
  theme(legend.position="bottom") #similar pattern in age distrubtion by survival
s2<-ggplot(f_titanic,aes(survived,age)) +
  stat_summary(geom="bar",fill="steelblue") +
  stat_summary(geom="errorbar")
s3<-ggplot(data=f_titanic,aes(fare,..density..)) + 
  geom_freqpoly(aes(color=survived)) +
  theme(legend.position="bottom") #similar to age
s4<-ggplot(f_titanic,aes(survived,fare)) +
  stat_summary(geom="bar",fill="steelblue") +
  stat_summary(geom="errorbar")
plot_grid(s1,s2,s3,s4,ncol=2)


#TABLES
survived_pclass_table<-
  f_titanic %>%
  group_by(survived,pclass) %>%
  summarize(
    N=n(),
    .groups="drop"
  )
survived_pclass_table
kable(survived_pclass_table,format="html") %>%
  kable_styling("striped","bordered") %>%
  add_header_above(c(" "=1,"Passenger survival by pclass"=2)) #title spans cols 2 and 3
