library(tidyverse) 
library(data.table)
library(rstudioapi)
library(skimr)
library(inspectdf)
library(mice)
library(plotly)
library(highcharter)
library(recipes) 
library(caret) 
library(purrr) 
library(graphics) 
library(Hmisc) 
library(glue)
library(h2o)
path <- dirname(getSourceEditorContext()$path)
path
df <- fread("C:/Users/ASUS/Downloads/Bootcamp 7/R Week 4/Day 2 ML LR/crimes.csv")
df
df %>% inspect_na()
df %>% glimpse()
df %>% colnames()
names(df) <- gsub("([a-z])([A-Z])", "\\1 \\2", df %>% colnames())
names(df) <- str_replace_all(names(df), ' ', '_') %>% str_to_lower()
names(df)
df <- df %>% select(violent_crimes_per_pop, everything())
df


library(scales)
df$pct_empl_prof_serv <- squish(df$pct_empl_prof_serv, quantile(df$pct_empl_prof_serv, c(.25, .75)))

df$pct_occup_manu <- squish(df$pct_occup_manu, quantile(df$pct_occup_manu, c(.25, .75)))

df$pct_mgmt_prof <- squish(df$pct_occup_mgmt_prof, quantile(df$pct_occup_mgmt_prof, c(.25, .75)))

df$male_pct_divorce <- squish(df$male_pct_divorce, quantile(df$male_pct_divorce, c(.25, .75)))

df$male_pct_nev_marr <- squish(df$male_pct_nev_marr, quantile(df$male_pct_nev_marr, c(.25, .75)))

df$pers_per_fam <- squish(df$pers_per_fam, quantile(df$pers_per_fam, c(.25, .75)))

df$pct_fam2par <- squish(df$pct_fam2par, quantile(df$pct_fam2par, c(.25, .75)))

df$pct_kids2par <- squish(df$pct_kids2par, quantile(df$pct_kids2par, c(.25, .75)))

df$pct_young_kids2par <- squish(df$pct_young_kids2par, quantile(df$pct_young_kids2par, c(.25, .75)))

df$pct_teen2par <- squish(df$pct_teen2par, quantile(df$pct_teen2par, c(.25, .95)))

df$pct_work_mom_young_kids <- squish(df$pct_work_mom_young_kids, quantile(df$pct_work_mom_young_kids, c(.25, .75)))

df$pct_work_mom <- squish(df$pct_work_mom, quantile(df$pct_work_mom, c(.25, .75)))

df$num_illeg <- squish(df$num_illeg, quantile(df$num_illeg, c(.25, .75)))

df$num_immig <- squish(df$num_immig, quantile(df$num_immig, c(.25, .75)))

df$pct_immig_recent <- squish(df$pct_immig_recent, quantile(df$pct_immig_recent, c(.25, .75)))

df$pct_immig_rec5 <- squish(df$pct_immig_rec5, quantile(df$pct_immig_rec5, c(.25, .75)))

df$pct_illeg <- squish(df$pct_illeg, quantile(df$pct_illeg, c(.25, .75)))

df$pct_occup_mgmt_prof <- squish(df$pct_occup_mgmt_prof, quantile(df$pct_occup_mgmt_prof, c(.25, .75)))
num_vars <- df %>% 
  select_if(is.numeric) %>% select(-violent_crimes_per_pop) %>% 
  names()
num_vars

library(graphics)
for (b in num_vars) {
  OutVals <- boxplot(df[[b]])$out
  if(length(OutVals)>0){
    print(paste0("----",b))
    print(OutVals)
  }
}
boxplot.stats(df$pct_immig_recent)$out

#Multicollinearity
target <- 'violent_crimes_per_pop'
features <- df %>% select(-violent_crimes_per_pop) %>% names()
features
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
f
glm <- glm(f, data = df)

glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df)

glm %>% summary() 
# VIF (Variance Inflation Factor) ----
while(glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[1] >= 1.5){
  afterVIF <- glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[-1] %>% names()
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df)
}

glm %>% faraway::vif() %>% sort(decreasing = T) %>% names() -> features 
df <- df %>% select(violent_crimes_per_pop,features)
# Standardize (Normalize) ----

new_df <- df[,-1]
new_df
new_df <- new_df %>% 
  scale() %>% as.data.frame()
new_df
df <- cbind(df[,1], new_df)
df %>% glimpse()
#Modeling
h2o.init()

h2o_data <- df %>% as.h2o()
#Splitting the data
h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'violent_crimes_per_pop'
features <- df %>% select(-violent_crimes_per_pop) %>% names()
# Fitting h2o model ----
model <- h2o.glm(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  nfolds = 10, seed = 123,
  lambda=0, compute_p_values = T)

model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>%
  .[-1,] %>%
  arrange(desc(p_value))
# Stepwise Backward Elimination ----
while(model@model$coefficients_table %>%
      as.data.frame() %>%
      dplyr::select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] > 0.05) {
  model@model$coefficients_table %>%
    as.data.frame() %>%
    dplyr::select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  test_h2o <- test %>% as.data.frame() %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target,
    training_frame = train,
    validation_frame = test,
    nfolds = 10, seed = 123,
    lambda = 0, compute_p_values = T)
}
model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) 

# Predicting the Test set results ----
y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred$predict
#Model Evaluation
test_set <- test %>% as.data.frame()
residuals = test_set$violent_crimes_per_pop - y_pred$predict
# Calculate RMSE (Root Mean Square Error) ----
RMSE = sqrt(mean(residuals^2))

# Calculate Adjusted R2 (R Squared) ----
y_test_mean = mean(test_set$violent_crimes_per_pop)

tss = sum((test_set$violent_crimes_per_pop - y_test_mean)^2) #total sum of squares
rss = sum(residuals^2) #residual sum of squares

R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))

tibble(RMSE = round(RMSE,1),
       R2, Adjusted_R2)
# Plotting actual & predicted ----
my_data <- cbind(predicted = y_pred$predict,
                 observed = test_set$violent_crimes_per_pop) %>% 
  as.data.frame()

g <- my_data %>% 
  ggplot(aes(predicted, observed)) + 
  geom_point(color = "darkred") + 
  geom_smooth(method=lm) + 
  labs(x="Predecited Power Output", 
       y="Observed Power Output",
       title=glue('Test: Adjusted R2 = {round(enexpr(Adjusted_R2),2)}')) +
  theme(plot.title = element_text(color="darkgreen",size=16,hjust=0.5),
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14), 
        axis.title.y = element_text(size=14))

g %>% ggplotly()

# Check overfitting ----
y_pred_train <- model %>% h2o.predict(newdata = train) %>% as.data.frame()

train_set <- train %>% as.data.frame()
residuals = train_set$violent_crimes_per_pop - y_pred_train$predict

RMSE_train = sqrt(mean(residuals^2))
y_train_mean = mean(train_set$violent_crimes_per_pop)

tss = sum((train_set$violent_crimes_per_pop - y_train_mean)^2)
rss = sum(residuals^2)

R2_train = 1 - (rss/tss); R2_train

n <- train_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2_train = 1-(1-R2_train)*((n-1)/(n-k-1))


# Plotting actual & predicted
my_data_train <- cbind(predicted = y_pred_train$predict,
                       observed = train_set$violent_crimes_per_pop) %>% 
  as.data.frame()

g_train <- my_data_train %>% 
  ggplot(aes(predicted, observed)) + 
  geom_point(color = "darkred") + 
  geom_smooth(method=lm) + 
  labs(x="Predecited Power Output", 
       y="Observed Power Output",
       title=glue('Train: Adjusted R2 = {round(enexpr(Adjusted_R2_train),2)}')) +
  theme(plot.title = element_text(color="darkgreen",size=16,hjust=0.5),
        axis.text.y = element_text(size=12), 
        axis.text.x = element_text(size=12),
        axis.title.x = element_text(size=14), 
        axis.title.y = element_text(size=14))

g_train %>% ggplotly()


# Compare 
library(patchwork)
g_train + g

tibble(RMSE_train = round(RMSE_train,1),
       RMSE_test = round(RMSE,1),
       
       Adjusted_R2_train,
       Adjusted_R2_test = Adjusted_R2)