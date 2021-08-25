#load packages
library(readr)
library(here)
library(rsample)
library(dplyr)
library(recipes)
library(parsnip)
library(kernlab)
library(workflows)
library(yardstick)
library(tune)
library(janitor)
library(forcats)
library(visdat)
library(stringr)
library(skimr)
library(vip)


#IV. Data modeling
#****************
#import cleaned and tidied data as a .csv to a tibble
f_titanic<-read_csv(here("data","tidy_data","tidy_train.csv"),
         col_types="fiffnnfffff")

#re-level factors
f_titanic$pclass<-fct_relevel(f_titanic$pclass,c("1","2","3"))
f_titanic$ticket_cat<-fct_relevel(f_titanic$ticket_cat,c("under_10k",
                                 "10k_up",
                                 "100k",
                                 "200k",
                                 "300k",
                                 "mil",
                                 "A",
                                 "C",
                                 "P",
                                 "SC",
                                 "SOT",
                                 "STO",
                                 "other"))

#1-create cross-validation samples with rsample
set.seed(27)
vfold_titanic<-vfold_cv(data=f_titanic,v=4)
vfold_titanic
pull(vfold_titanic,splits)


#2-create recipe
titanic_recipe<-recipe(f_titanic) %>%
  update_role(passenger_id,new_role="id variable") %>% 
  update_role(pclass:fam_size,new_role="predictor") %>% 
  update_role(survived,new_role="outcome")
titanic_recipe


#3-use parsnip() to specify model
#logistic model
titanic_mod_log<-logistic_reg() %>%
  set_mode("classification") %>%
  set_engine("glm") 
titanic_mod_log

#svm m
titanic_mod_svm<-svm_poly() %>%
  set_mode("classification") %>%
  set_engine("kernlab")
titanic_mod_svm


#4-make a workflow
#logistic model
titanic_mod_log_wflow<-workflow() %>%
  add_recipe(titanic_recipe) %>%
  add_model(titanic_mod_log)
titanic_mod_log_wflow

#svm 
titanic_mod_svm_wflow<-workflow() %>%
  add_recipe(titanic_recipe) %>%
  add_model(titanic_mod_svm)
titanic_mod_svm_wflow


#5-fit model
titanic_mod_log_wflow_fit<-parsnip::fit(titanic_mod_log_wflow,data=f_titanic) #log reg

titanic_mod_svm_wflow_fit<-parsnip::fit(titanic_mod_svm_wflow,data=f_titanic) #svm


#6-get predicted values using predict() and assess their accuracy
pred_titanic_log<-predict(titanic_mod_log_wflow_fit,new_data=f_titanic) #log reg

pred_titanic_svm<-predict(titanic_mod_svm_wflow_fit,new_data=f_titanic) #svm


#7-compare predicted and actual variables
accuracy(f_titanic,
         truth=survived,estimate=pred_titanic_log$.pred_class) #82.8% accurate
count(f_titanic,survived) #549-342 dead-survived (actual)
count(pred_titanic_log,.pred_class) #580-311 dead-survived (predicted)

accuracy(f_titanic,
         truth=survived,estimate=pred_titanic_svm$.pred_class) #81.0% accurate
count(f_titanic,survived) #549-342 dead-survived (actual)
count(pred_titanic_svm,.pred_class) #588-303 dead-survived (predicted)


#8-fit model to cross-validation folds
set.seed(523)
resample_log_fit<-fit_resamples(titanic_mod_log_wflow,vfold_titanic)
collect_metrics(resample_log_fit) #81.1% accurate

set.seed(523)
resample_svm_fit<-fit_resamples(titanic_mod_svm_wflow,vfold_titanic)
collect_metrics(resample_svm_fit) #79.4% 


#9-model tuning
#logistic regression model
titanic_mod_log_tune<-logistic_reg(penalty=tune(),mixture=tune()) %>%
  set_mode("classification") %>%
  set_engine("glm")

titanic_mod_log_wflow_tune<-workflow() %>%
  add_recipe(titanic_recipe) %>%
  add_model(titanic_mod_log_tune)

resample_log_fit2<-tune_grid(titanic_mod_log_wflow_tune,resamples=vfold_titanic,grid=30) #not tuning
collect_metrics(resample_log_fit2) #81.1%

#svm m
titanic_mod_svm_tune<-svm_poly(cost=tune(),degree=tune(),scale_factor=tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

titanic_mod_svm_wflow_tune<-workflow() %>%
  add_recipe(titanic_recipe) %>%
  add_model(titanic_mod_svm_tune)

set.seed(95)
resample_svm_fit2<-tune_grid(titanic_mod_svm_wflow_tune,resamples=vfold_titanic,grid=30) 
collect_metrics(resample_svm_fit2)
show_best(resample_svm_fit2,metric="accuracy") #78.6%; cost=23, degree=2.49, scale_factor=.000215
#chose logistic regression model due to higher accuracy on cross-validation dataset

#10-select best model
tuned_log_values<-select_best(resample_log_fit2,"accuracy")


#11-finalize workflow and model using tuned values
tuned_log_wflow<-titanic_mod_log_wflow_tune %>%
  finalize_workflow(tuned_log_values)
tuned_log_wflow
fitted_tuned_log_mod<-fit(tuned_log_wflow,f_titanic) %>% pull_workflow_fit()


#12-model diagnostics
log_mod_coef<-enframe(fitted_tuned_log_mod$fit$coefficients)
log_mod_coef %>% 
  rename(coef="value") %>%
  arrange(desc(coef)) %>%
  print(n=30)
vi(fitted_tuned_log_mod)
vip(fitted_tuned_log_mod)

#VI. Assess model performance on test data
#*****************************************
#Read in data
t_titanic<-read_csv(here("data","raw_data","test.csv"),
                    col_types="ifcfniicncf")
t_titanic


#Clean data
#1. Preliminary data checking
nrow(t_titanic); ncol(t_titanic) #check # of rows/cols
str(t_titanic) #check classes of variables
head(t_titanic,n=10); tail(t_titanic,n=10) #check top/bottom of tibble


#2. Data cleaning
t_titanic<-clean_names(t_titanic) #clean names
#re-level factors to match training set
t_titanic$pclass<-fct_relevel(t_titanic$pclass,c("1","2","3"))
t_titanic$embarked<-fct_relevel(t_titanic$embarked,c("C","Q","S"))


#3. Data imputation
#assess missing data
vis_dat(t_titanic)
vis_miss(t_titanic)
#lots of cabin data missing, about 20% age data missing, and 1 fare data point missing

#impute data
t_titanic$cabin[is.na(t_titanic$cabin)]<-"unknown_other" #replace NAs with unknown_other
#use same preprocessing from training data
imputed<-readRDS(here("data","tidy_data","imputed_data_recipe"))
it_titanic<-prep(imputed) %>% bake(new_data=t_titanic)
summary(it_titanic) #no missing data
ct_titanic<-bind_cols(t_titanic[,1],it_titanic[,1:7],t_titanic[,c(3,8,10)])


#4. Data checking
#check n's (using prior knowledge)
range(ct_titanic$age,na.rm=T) #0.17-76
range(ct_titanic$fare,na.rm=T) #0-512
range(ct_titanic$sib_sp) #0-8
range(ct_titanic$parch) #0-9
#all seem reasonable

#validate with external data
#from wiki: 24.6% 1st class; 21.5% 2nd class; and 53.8% 3rd class
tabyl(ct_titanic,pclass) #25.6%, 22.2%, and 52.1% (seem close)

#from wiki: 66% male and 34% female
tabyl(ct_titanic,sex) #63.6% m and 36.4% f (again, close)

#data summaries (with imputed data) 
summary(ct_titanic)
skim(ct_titanic)


#5. Feature engineering
#cabin: code as factor
sort(ct_titanic$cabin)
length(unique(ct_titanic$cabin)) #77 different cabin types
ct_titanic$cabin[str_which(ct_titanic$cabin,"^T|^G")]<-"unknown_other" 
#replaces T & G cabinswith u_o (few numbers)

#bin cabin names by first letter into types
ct_titanic<-ct_titanic %>%
  mutate(cabin_type=case_when(
    str_detect(cabin,"^A")~"A",
    str_detect(cabin,"^B")~"B",
    str_detect(cabin,"^C")~"C",
    str_detect(cabin,"^D")~"D",
    str_detect(cabin,"^E")~"E",
    str_detect(cabin,"^F")~"F",
    str_detect(cabin,"unknown_other")~"UO"
  )) 
tail(ct_titanic[,11:12],10) #check, and it works
sum(is.na(ct_titanic$cabin_type)) #0; no NAs
ct_titanic$cabin_type<-as.factor(ct_titanic$cabin_type) #makes it a factor
levels(ct_titanic$cabin_type) #7 levels


#tickets: code as factor
#bin tickets into categories (as a factor)
sort(ct_titanic$ticket) #sort tickets by character
sort(as.numeric(ct_titanic$ticket)) #sort tickets by number

ct_titanic$ticket<-
  str_replace_all(ct_titanic$ticket,"[[:punct:]]","") #removes punctuation

#bins ticket prefixes into categories
ct_titanic<-ct_titanic %>% 
  mutate(ticket_cat=case_when(
    as.numeric(ticket)<10000~"under_10k",
    between(as.numeric(ticket),10000,100000)~"10k_up",
    between(as.numeric(ticket),100000,200000)~"100k",
    between(as.numeric(ticket),200000,300000)~"200k",
    between(as.numeric(ticket),300000,1000000)~"300k",
    as.numeric(ticket)>1000000~"mil",
    str_detect(ticket,"^A")~"A",
    str_detect(ticket,"^C")~"C",
    str_detect(ticket,"^P")~"P",
    str_detect(ticket,"^SC")~"SC",
    str_detect(ticket,"^SOT")~"SOT",
    str_detect(ticket,"^S(TO)")~"STO",
    str_detect(ticket,"^SO(C|P)")~"other",
    str_detect(ticket,"^W|^F|^L|^SP|^SW")~"other"
  )) 
unique(ct_titanic$ticket_cat)
sum(is.na(ct_titanic$ticket_cat)) #0; no NAs
ct_titanic$ticket_cat<-as.factor(ct_titanic$ticket_cat) #makes it a factor
levels(ct_titanic$ticket_cat)<-c("under_10k",
                                 "10k_up",
                                 "100k",
                                 "200k",
                                 "300k",
                                 "mil",
                                 "A",
                                 "C",
                                 "P",
                                 "SC",
                                 "SOT",
                                 "STO",
                                 "other")
levels(ct_titanic$ticket_cat)


#marital status: extract fromm name variable 
#variable creation (for marital status): unk_na=male & unk f; Fm=married female; Fum: unmarried female
ct_titanic<-ct_titanic %>%
  mutate(marital_status=case_when(
    sex=="male"~"unk_na",
    str_detect(name,"Mrs\\.|Mme\\.|Lady\\.(.*)Mrs")~"Fm",
    sex=="female" & str_detect(name,"Dr\\.")~"unk_na",
    str_detect(name,"Countess\\.|Ms.|Dona\\.")~"unk_na",
    str_detect(name,"Miss\\.|Mlle\\.")~"Fum"
  ))

unique(ct_titanic$marital_status) #3 categories
sum(is.na(ct_titanic$marital_status)) #0; no NAs
ct_titanic$marital_status<-as.factor(ct_titanic$marital_status) #makes it a factor
levels(ct_titanic$marital_status) #same 3 categories
tabyl(ct_titanic$marital_status)


#family size: combine parch and sib_sp
ct_titanic<-ct_titanic %>% mutate(fam_size=parch+sib_sp)
ct_titanic$fam_size<-as.factor(ct_titanic$fam_size)
ct_titanic$fam_size<-fct_collapse(ct_titanic$fam_size,solo="0",small=c("1","2","3"),
                                 large=c("4","5","6","7","8","9","10","11","12","13","14","15","16","17",
                                         "18","19","20"))


#remove unnecessary columns and convert to ft_titanic
ct_titanic
ft_titanic<-ct_titanic[,-c(5:6,9:11)] #removes sib_sp, parch, name, ticket, and cabin
ft_titanic


#Predict survival of test dataset
pred_titanic<-predict(fitted_tuned_log_mod,new_data=ft_titanic)
pred_titanic
dim(pred_titanic)

kpost_titanic_submit<-bind_cols(ft_titanic$passenger_id,pred_titanic[,1])
kpost_titanic_submit<-rename(kpost_titanic_submit,PassengerID="...1",Survived=".pred_class")
kpost_titanic_submit #0.76794


#Export predictions to .csv
write_csv(kpost_titanic_submit,here("data","predictions","kpost_titanic_submission.csv"))
