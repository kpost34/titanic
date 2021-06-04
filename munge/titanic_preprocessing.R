#load packages
library(here)
library(readr)
library(visdat)
library(dplyr)
library(skimr)
library(janitor)
library(forcats)
library(recipes)
library(ggplot2)
library(stringr)


#I. Data import
#**************
titanic<-read_csv(here("data","raw_data","train.csv"),
                  col_types="iffcfniicncf")
titanic



#II. Data cleaning, wrangling, and preprocessing
#***********************************************
#1. Preliminary data checking
nrow(titanic); ncol(titanic) #check # of rows/cols
str(titanic) #check classes of variables
head(titanic,n=10); tail(titanic,n=10) #check top/bottom of tibble


#2. Data cleaning
titanic<-clean_names(titanic) #clean names
titanic<-relocate(titanic,survived,before=passenger_id) #rearranges cols
levels(titanic$pclass) #not in numerical order
titanic$pclass<-fct_relevel(titanic$pclass,c("1","2","3"))


#3. Data imputation
#assess missing data
vis_dat(titanic)
vis_miss(titanic)
#lots of cabin data missing, about 20% age data missing, and 2 emabrked data missing

#impute data
titanic$cabin[is.na(titanic$cabin)]<-"unknown_other" #replace NAs with unknown_other
imputed<-recipe(survived~pclass + sex + age + sib_sp + parch + fare + embarked,data=titanic) %>%
  step_knnimpute(age,embarked,fare,neighbors=10) #via k-nearest neighbors
saveRDS(imputed,here("data","tidy_data","imputed_data_recipe"))
i_titanic<-prep(imputed) %>% bake(new_data=titanic)
summary(i_titanic) #no missing data
c_titanic<-bind_cols(titanic[,c(1,2)],i_titanic[,1:7],titanic[,c(4,9,11)])


#4. Data checking
#check n's (using prior knowledge)
range(c_titanic$age,na.rm=T) #0.42-80
range(c_titanic$fare,na.rm=T) #0-512
range(c_titanic$sib_sp) #0-8
range(c_titanic$parch) #0-6
#all seem reasonable

#validate with external data
#from wiki: 24.6% 1st class; 21.5% 2nd class; and 53.8% 3rd class
tabyl(c_titanic,pclass) #24.2%, 20.7%, and 55.1% (seem close)

#from wiki: 66% male and 34% female
tabyl(c_titanic,sex) #64.7% m and 35.2% f (again, close)

#data summaries (with imputed data)
summary(c_titanic)
skim(c_titanic)


#5. Feature engineering
#cabin: code as factor
sort(c_titanic$cabin)
length(unique(c_titanic$cabin)) #148 different cabin types
unique(c_titanic$cabin)[1:15] #overlapping prefixes (which indicate cabin & possibly
#survival)
c_titanic$cabin[str_which(c_titanic$cabin,"^T|^G")]<-"unknown_other" 
#replaces T & G cabinswith u_o (few numbers)

#bin cabin names by first letter into types
c_titanic<-c_titanic %>%
  mutate(cabin_type=case_when(
    str_detect(cabin,"^A")~"A",
    str_detect(cabin,"^B")~"B",
    str_detect(cabin,"^C")~"C",
    str_detect(cabin,"^D")~"D",
    str_detect(cabin,"^E")~"E",
    str_detect(cabin,"^F")~"F",
    str_detect(cabin,"unknown_other")~"UO"
  )) 
head(c_titanic[,12:13],10) #check, and it works
tabyl(c_titanic$cabin_type) #cabin_type level with lowest obs is F with 13
sum(is.na(c_titanic$cabin_type)) #0; no NAs
c_titanic$cabin_type<-as.factor(c_titanic$cabin_type) #makes it a factor
levels(c_titanic$cabin_type) #7 levels

tabyl(c_titanic,survived,cabin_type) #indicates association between some cabin types and survival


#tickets: code as factor
#bin tickets into categories (as a factor)
sort(c_titanic$ticket) #sort tickets by character
sort(as.numeric(c_titanic$ticket)) #sort tickets by number

c_titanic$ticket<-
  str_replace_all(c_titanic$ticket,"[[:punct:]]","") #removes punctuation

#preview some of the ticket prefixes
filter(c_titanic,str_detect(ticket,"^STO"))
filter(c_titanic,str_detect(ticket,"^SOT"))
filter(c_titanic,str_detect(ticket,"^S")) %>% print(n=65)
filter(c_titanic,str_detect(ticket,"^SO(C|P)")) %>% print(n=10)

#look more closely at ticket number (based on number ranges) and survival
c_titanic %>% filter(as.numeric(ticket)<10000) %>% tabyl(survived) #less than 10k; 36% survival
c_titanic %>% filter(between(as.numeric(ticket),10000,100000)) %>% tabyl(survived) #b/t 10k & 100k; 62% survival
c_titanic %>% filter(between(as.numeric(ticket),100000,200000)) %>% tabyl(survived) #b/t 100 & 200k; 51% survival
c_titanic %>% filter(between(as.numeric(ticket),200000,300000)) %>% tabyl(survived) #b/t 200 & 300k; 48% survival
c_titanic %>% filter(between(as.numeric(ticket),100000,1000000)) %>% tabyl(survived) #b/t 100k & 1 mil.; 32% survival
#note: bin all 5-digit ticket numbers then you lose effect of 300k range
c_titanic %>% filter(between(as.numeric(ticket),300000,1000000)) %>% tabyl(survived) #b/t 300k & 1 mil.; 21% survivalk
c_titanic %>% filter(as.numeric(ticket)>1000000) %>% tabyl(survived) #greater than 1 mil.; 25% survival
#appears to be association between ticket number and survival

#bins ticket prefixes into categories
c_titanic<-c_titanic %>% 
  mutate(ticket_cat=case_when(
    as.numeric(ticket)<10000~"under_10k",
    between(as.numeric(ticket),10000,100000)~"10k_up",
    between(as.numeric(ticket),100000,200000)~"100k",
    between(as.numeric(ticket),200000,300000)~"200k",
    between(as.numeric(ticket),300000,1000000)~"300k",
    as.numeric(ticket)>1000000~"mil",
    str_detect(ticket,"^A")~"A",
    str_detect(ticket,"^C")~"C",
    str_detect(ticket,"^F")~"F",
    str_detect(ticket,"^P")~"P",
    str_detect(ticket,"^SC")~"SC",
    str_detect(ticket,"^SO(C|P)")~"SO",
    str_detect(ticket,"^SOT")~"SOT",
    str_detect(ticket,"^S(TO)")~"STO",
    str_detect(ticket,"^W")~"W",
    str_detect(ticket,"^L|^SP|^SW")~"other"
  )) 
unique(c_titanic$ticket_cat)
sum(is.na(c_titanic$ticket_cat)) #0; no NAs
c_titanic$ticket_cat<-as.factor(c_titanic$ticket_cat) #makes it a factor
levels(c_titanic$ticket_cat)
c_titanic$ticket_cat<-fct_relevel(c_titanic$ticket_cat,
                                  "under_10k",
                                  "10k_up",
                                  "100k",
                                  "200k",
                                  "300k",
                                  "mil",
                                  "A",
                                  "C",
                                  "F",
                                  "P",
                                  "SC",
                                  "SO",
                                  "SOT",
                                  "STO",
                                  "W",
                                  "other")
levels(c_titanic$ticket_cat)                           
tabyl(c_titanic,survived,ticket_cat) #clearly some ticket numbers/prefixes are associated with survival

#graph relationship between survival and ticket category
c_titanic %>%
  ggplot() +
  geom_bar(aes(ticket_cat,fill=survived),position="fill") +
  scale_fill_manual(values=c("darkred","dodgerblue"))
#6 categories are highly discriminating


#special titles: extract from name variable
#teasing out names--higher status (master, dr, )
#exploration: males
str_subset(c_titanic$name,"Mr\\.") #517
str_subset(c_titanic$name,"Col\\.") #colonels; 2
str_subset(c_titanic$name,"Master") #masters; 40
str_subset(c_titanic$name,"Dr\\.") #doctors; 6 male (7 total)
str_subset(c_titanic$name,"Rev\\.") #reverends; 6
str_subset(c_titanic$name,"Major\\.") #majors; 2
str_subset(c_titanic$name,"Capt\\.") #captain; 1
str_subset(c_titanic$name,"Sir\\.") #knights?; 1
517+2+40+6+6+2+1+1+1 #576 (+ 1 without title; Uruchurtu, Don. Manuel E)
filter(c_titanic,sex=="male") #577
#females
str_subset(c_titanic$name,"Lady\\.") #1
str_subset(c_titanic$name,"Jonkheer\\.") #1
str_subset(c_titanic$name,"Countess\\.") #1
str_subset(c_titanic$name,"Dona\\.") #0

#variable creation (for special titles): 1=yes; 0=no
c_titanic<-c_titanic %>%
  mutate(spec_title=if_else(
    str_detect(name,"Mr\\.|Col\\.|Master\\.|Dr\\.|Rev\\.|Major\\.
               |Capt\\.|Sir\\.|Lady\\.|Jonkheer\\.|Countess\\.|Dona\\."),
    1,0))
c_titanic$spec_title<-as.factor(c_titanic$spec_title)

tabyl(c_titanic,survived,spec_title) #table of survival and title data: lower survival associated with st

#graph of title split on survival
c_titanic %>%
  ggplot() +
  geom_bar(aes(spec_title,fill=survived),position="fill") +
  scale_fill_manual(values=c("purple4","darkgreen"))


#marital status: extract fromm name variable 
#test of coding 
str_subset(c_titanic$name,"Mrs") #129
str_subset(c_titanic$name,"Mrs\\.") #married women; 125
str_subset(c_titanic$name,"Mme\\.") #married woman; 1
str_subset(c_titanic$name,"Lady\\.(.*)Mrs") #married woman; 1
str_subset(c_titanic$name,"Dr\\.") #1 female doctor; unknown marital status
str_subset(c_titanic$name,"Countess\\.") #1 countess; unknown marital status
str_subset(c_titanic$name,"Ms\\.") #unknown marital status; 1
str_subset(c_titanic$name, "Miss\\.") #unmarried women; 182
str_subset(c_titanic$name,"Mlle\\.") #unmarried women; 2
125+1+1+1+1+1+182+2 #314
filter(c_titanic,sex=="female") #314

#variable creation (for female marital status): M=male; Fm=married female; Funk: female unk mar status; Fum: 
#unmarried female
#combined with sex variable by adding levels because m status did not divide males
c_titanic<-c_titanic %>%
  mutate(sex_marital_status=case_when(
    sex=="male"~"M",
    str_detect(name,"Mrs\\.|Mme\\.|Lady\\.(.*)Mrs")~"Fm",
    sex=="female" & str_detect(name,"Dr\\.")~"Funk",
    str_detect(name,"Countess\\.|Ms.|Dona\\.")~"Funk",
    str_detect(name,"Miss\\.|Mlle\\.")~"Fum"
  ))

unique(c_titanic$sex_marital_status) #4 categories
sum(is.na(c_titanic$sex_marital_status)) #0; no NAs
c_titanic$sex_marital_status<-as.factor(c_titanic$sex_marital_status) #makes it a factor
levels(c_titanic$sex_marital_status) #same 4 categories

#comparison between two variables
tabyl(c_titanic,survived,sex_marital_status) 
tabyl(c_titanic,survived,sex) 
#disparity in survival between unmarried and married women (perhaps confounded by age)

#remove unnecessary columns and convert to f_titanic--WILL NEED TO CHANGE LATER
c_titanic
f_titanic<-c_titanic[,-c(4,10:12)] #remove sex, name, ticket, and cabin
f_titanic


#export tibble as a .csv
write_csv(f_titanic,here("data","tidy_data","tidy_train.csv"))
