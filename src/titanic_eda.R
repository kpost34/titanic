#load packages
library(readr)
library(here)
library(skimr)
library(dplyr)
library(ggplot2)
library(cowplot)
library(knitr)
library(kableExtra)
library(janitor)


#III. Data exploration/visualization
#**********************************
#import cleaned and tidied data as a .csv to a tibble
f_titanic<-read_csv(here("data","tidy_data","tidy_train.csv"),
         col_types="fiffnnfffff")


#Summary stats 
summary(f_titanic)
skim(f_titanic) 


#Predictors only
#Univariate
#categorical
ggplot(data=f_titanic)+geom_bar(aes(survived)) #~550 Died & ~350 Survived
ggplot(data=f_titanic)+geom_bar(aes(pclass)) #almost 2.5x class 3 than 1 or 2
ggplot(data=f_titanic)+geom_bar(aes(embarked)) #~2x C than Q and ~4x S than C 
ggplot(data=f_titanic)+geom_bar(aes(fam_size)) #~600 0, ~300 small, and ~ 50 large
#most passengers traveled alone

#continuous
p1<-ggplot(data=f_titanic)+geom_histogram(aes(age),binwidth=2) #right-skewed normal with a valley ~10 yo and higher than expected <1 yo
p2<-ggplot(data=f_titanic)+geom_density(aes(age))
p3<-ggplot(data=f_titanic)+geom_histogram(aes(fare)) #fares generally under $275 except for at least one >$500; resembles Poisson distribution
p4<-ggplot(data=f_titanic)+geom_density(aes(fare))
plot_grid(p1,p2,p3,p4,ncol=2)
#age is close to normal except more newborns/toddlers and fewer teens than expected 
#perhaps fare could be modeled with a log-normal or gamma distrubition; notice a couple fares > $500


#Bivariate
#categorical-continuous
ggplot(data=f_titanic) +
  geom_boxplot(aes(pclass,age))
 #age by pclass: age tends to decrease as pclass increases

#categorical/integer-continuous or cotinuous-continuous
ggplot(data=f_titanic)+geom_point(aes(age,fare),method="lm") #fare outliers (>$500) for two people just shy of 40 yo; no obvious relationship here
f_titanic %>% filter(fare<500) %>% ggplot()+geom_point(aes(age,fare))+geom_smooth(aes(age,fare),method="lm") #here it is without the high fares
f_titanic %>% filter(fare>500)

ggplot(data=f_titanic)+geom_point(aes(age,fare,color=pclass)) 
#lower class passengers generally had lower ticket fares at each age
ggplot(data=f_titanic,aes(fare,..density..))+geom_freqpoly()+facet_wrap(~ fam_size) 
#similar pattern among all fam_size levels (peaks at very low fare) except that small fam_size indicates a smaller peak and more observations at fares <200
ggplot(data=f_titanic,aes(fare,..density..))+geom_freqpoly(aes(color=embarked))


#Predictors and dependent variable (survived)
#categorical-continuous
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

#without high fares
f_titanic %>%
  filter(fare<=500) %>%
  ggplot(aes(survived,fare)) +
  stat_summary(geom="bar",fill="steelblue") +
  stat_summary(geom="errorbar")


#TABLE
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