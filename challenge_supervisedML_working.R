# Instalar rstanarm si no está instalado
if (!requireNamespace("rstanarm", quietly = TRUE)) {
  install.packages("rstanarm")
}

# Cargar librerías
library(tidymodels)  # for the parsnip package, along with the rest of tidymodels
library(broom.mixed) # for converting bayesian models to tidy tibbles
library(rstanarm)   # para el modelo bayesiano

# Datos
bike_data_tbl <- readRDS("raw_data/bike_orderlines.rds")

# Filtrar los datos para excluir la categoría "Gravel"
bike_data_filtered_tbl <- bike_data_tbl %>% 
  filter(category_1 != "Gravel")

# Distribución previa
prior_dist <- rstanarm::student_t(df = 1)

set.seed(123)

# Crear el modelo con parsnip
bayes_mod <- linear_reg() %>% 
  set_engine("stan", 
             prior_intercept = prior_dist, 
             prior = prior_dist)

# Entrenar el modelo
bayes_fit <- bayes_mod %>% 
  fit(weight ~ price * category_1, 
      data = bike_data_filtered_tbl)

print(bayes_fit, digits = 5)

# Extraer las predicciones
predictions <- predict(bayes_fit, bike_data_filtered_tbl, type = "numeric")

# Añadir las predicciones al conjunto de datos
bike_data_filtered_tbl <- bike_data_filtered_tbl %>% 
  mutate(pred = predictions$.pred)

# Graficar resultados
ggplot(bike_data_filtered_tbl, aes(x = category_1)) +
  geom_point(aes(y = pred)) +
  geom_errorbar(aes(ymin = pred - 1.96 * predictions$.pred_se_fit, 
                    ymax = pred + 1.96 * predictions$.pred_se_fit), 
                width = .2) +
  labs(y = "Bike weight", x = "Category")

bayes_plot_data <- 
  new_points %>%
  bind_cols(predict(bayes_fit, new_data = new_points)) %>% 
  bind_cols(predict(bayes_fit, new_data = new_points, type = "conf_int"))

ggplot(bayes_plot_data, aes(x = category_1)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) + 
  labs(y = "Bike weight") + 
  ggtitle("Bayesian model with t(1) prior distribution")

# 2. Preprocessing ----
# Instalar y cargar los paquetes necesarios
if (!requireNamespace("nycflights13", quietly = TRUE)) {
  install.packages("nycflights13")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}
if (!requireNamespace("lubridate", quietly = TRUE)) {
  install.packages("lubridate")
}

library(nycflights13)
library(dplyr)
library(lubridate)

set.seed(123)

# Cargar y preparar los datos
flight_data <- flights %>%
  mutate(
    # Convertir el retraso en la llegada a un factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # Usar la fecha (no la fecha-hora) en la receta
    date = as.Date(time_hour)
  ) %>%
  # Incluir los datos meteorológicos
  inner_join(weather, by = c("origin", "time_hour")) %>%
  # Retener solo las columnas específicas que utilizaremos
  select(dep_time, flight, origin, dest, air_time, distance, 
         carrier, date, arr_delay, time_hour) %>%
  # Excluir datos faltantes
  na.omit() %>%
  # Convertir columnas cualitativas a factores
  mutate_if(is.character, as.factor)

# Ver las primeras filas del conjunto de datos preparado
head(flight_data)

flight_data %>% 
  count(arr_delay) %>% 
  mutate(prop = n/sum(n))
glimpse(flight_data)

# Fix the random numbers by setting the seed 
# This enables the analysis to be reproducible when random numbers are used 
set.seed(555)
# Put 3/4 of the data into the training set 
data_split <- initial_split(flight_data, prop = 3/4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)

flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") 

flight_data %>% 
  distinct(date) %>% 
  mutate(numeric_date = as.numeric(date)) 

flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors())
lr_mod <- 
  logistic_reg() %>% 
  set_engine("glm")

flights_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(flights_rec)
flights_wflow

flights_fit <- 
  flights_wflow %>% 
  fit(data = train_data)

flights_fit %>% 
  pull_workflow_fit() %>% 
  tidy()

predict(flights_fit, test_data)

flights_pred <- 
  predict(flights_fit, test_data, type = "prob") %>% 
  bind_cols(test_data %>% select(arr_delay, time_hour, flight)) 

# The data look like: 
flights_pred

flights_pred %>% 
  roc_curve(truth = arr_delay, .pred_late) %>% 
  autoplot()

flights_pred %>% 
  roc_auc(truth = arr_delay, .pred_late)

# 3.Evaluating ----
library(tidymodels) # for the rsample package, along with the rest of tidymodels

# Helper packages
library(modeldata)  # for the cells data
data(cells, package = "modeldata")
cells

cells %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)

cell_train <- training(cell_split)
cell_test  <- testing(cell_split)

nrow(cell_train)
nrow(cell_train)/nrow(cells)
cell_train %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))
cell_test %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

rf_mod <- 
  rand_forest(trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

set.seed(234)
rf_fit <- 
  rf_mod %>% 
  fit(class ~ ., data = cell_train)
rf_fit

rf_training_pred <- 
  predict(rf_fit, cell_train) %>% 
  bind_cols(predict(rf_fit, cell_train, type = "prob")) %>% 
  # Add the true outcome data back in
  bind_cols(cell_train %>% 
              select(class))

rf_training_pred %>%                # training set predictions
  roc_auc(truth = class, .pred_PS)

rf_training_pred %>%                # training set predictions
  accuracy(truth = class, .pred_class)

rf_testing_pred <- 
  predict(rf_fit, cell_test) %>% 
  bind_cols(predict(rf_fit, cell_test, type = "prob")) %>% 
  bind_cols(cell_test %>% select(class))

rf_testing_pred %>%                   # test set predictions
  roc_auc(truth = class, .pred_PS)

rf_testing_pred %>%                   # test set predictions
  accuracy(truth = class, .pred_class)

set.seed(345)
folds <- vfold_cv(cell_train, v = 10)
folds

rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_formula(class ~ .)

set.seed(456)
rf_fit_rs <- 
  rf_wf %>% 
  fit_resamples(folds)

rf_fit_rs

collect_metrics(rf_fit_rs)

rf_testing_pred %>%                   # test set predictions
  roc_auc(truth = class, .pred_PS)

#4. Tunning ----
library(tidymodels)  # for the tune package, along with the rest of tidymodels

# Helper packages
library(modeldata)   # for the cells data
library(vip)         # for variable importance plots

set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)
cell_train <- training(cell_split)
cell_test  <- testing(cell_split)

tune_spec <- 
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

tune_spec

tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 5)

tree_grid %>% 
  count(tree_depth)

set.seed(234)
cell_folds <- vfold_cv(cell_train)

set.seed(345)

tree_wf <- workflow() %>%
  add_model(tune_spec) %>%
  add_formula(class ~ .)

tree_res <- 
  tree_wf %>% 
  tune_grid(
    resamples = cell_folds,
    grid = tree_grid
  )

tree_res

tree_res %>% 
  collect_metrics()

tree_res %>%
  collect_metrics() %>%
  mutate(tree_depth = factor(tree_depth)) %>%
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

tree_res %>%
  show_best()

best_tree <- tree_res %>%
  select_best()

best_tree

final_wf <- 
  tree_wf %>% 
  finalize_workflow(best_tree)

final_wf

final_tree <- 
  final_wf %>%
  fit(data = cell_train) 

final_tree

library(vip)

final_tree %>% 
  pull_workflow_fit() %>% 
  vip()

final_fit <- 
  final_wf %>%
  last_fit(cell_split) 

final_fit %>%
  collect_metrics()
## # A tibble: 2 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.802
## 2 roc_auc  binary         0.860

final_fit %>%
  collect_predictions() %>% 
  roc_curve(class, .pred_PS) %>% 
  autoplot()

args(decision_tree)

# BUSINESS CASE ----
# Standard
library(tidyverse)

# Modeling
library(parsnip)

# Preprocessing & Sampling
library(recipes)
library(rsample)

# Modeling Error Metrics
library(yardstick)

# Plotting Decision Trees
library(rpart.plot)

# Modeling ----------------------------------------------------------------
bike_orderlines_tbl <- readRDS("raw_data/bike_orderlines.rds")
glimpse(bike_orderlines_tbl)

model_sales_tbl <- bike_orderlines_tbl %>%
  select(total_price, model, category_2, frame_material) %>%
  
  group_by(model, category_2, frame_material) %>%
  summarise(total_sales = sum(total_price)) %>%
  ungroup() %>%
  
  arrange(desc(total_sales))

model_sales_tbl %>%
  mutate(category_2 = as_factor(category_2) %>% 
           fct_reorder(total_sales, .fun = max) %>% 
           fct_rev()) %>%
  
  ggplot(aes(frame_material, total_sales)) +
  geom_violin() +
  geom_jitter(width = 0.1, alpha = 0.5, color = "#2c3e50") +
  #coord_flip() +
  facet_wrap(~ category_2) +
  scale_y_continuous(labels = scales::dollar_format(scale = 1e-6, suffix = "M", accuracy = 0.1)) +
  tidyquant::theme_tq() +
  labs(
    title = "Total Sales for Each Model",
    x = "Frame Material", y = "Revenue"
  )

bike_features_tbl <- readRDS("raw_data/bike_features_tbl.rds")
glimpse(bike_features_tbl)

bike_features_tbl <- bike_features_tbl %>% 
  select(model:url, `Rear Derailleur`, `Shift Lever`) %>% 
  mutate(
    `shimano dura-ace`        = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano dura-ace ") %>% as.numeric(),
    `shimano ultegra`         = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano ultegra ") %>% as.numeric(),
    `shimano 105`             = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano 105 ") %>% as.numeric(),
    `shimano tiagra`          = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano tiagra ") %>% as.numeric(),
    `Shimano sora`            = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano sora") %>% as.numeric(),
    `shimano deore`           = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano deore(?! xt)") %>% as.numeric(),
    `shimano slx`             = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano slx") %>% as.numeric(),
    `shimano grx`             = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano grx") %>% as.numeric(),
    `Shimano xt`              = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano deore xt |shimano xt ") %>% as.numeric(),
    `Shimano xtr`             = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano xtr") %>% as.numeric(),
    `Shimano saint`           = `Rear Derailleur` %>% str_to_lower() %>% str_detect("shimano saint") %>% as.numeric(),
    `SRAM red`                = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram red") %>% as.numeric(),
    `SRAM force`              = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram force") %>% as.numeric(),
    `SRAM rival`              = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram rival") %>% as.numeric(),
    `SRAM apex`               = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram apex") %>% as.numeric(),
    `SRAM xx1`                = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram xx1") %>% as.numeric(),
    `SRAM x01`                = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram x01|sram xo1") %>% as.numeric(),
    `SRAM gx`                 = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram gx") %>% as.numeric(),
    `SRAM nx`                 = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram nx") %>% as.numeric(),
    `SRAM sx`                 = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram sx") %>% as.numeric(),
    `SRAM sx`                 = `Rear Derailleur` %>% str_to_lower() %>% str_detect("sram sx") %>% as.numeric(),
    `Campagnolo potenza`      = `Rear Derailleur` %>% str_to_lower() %>% str_detect("campagnolo potenza") %>% as.numeric(),
    `Campagnolo super record` = `Rear Derailleur` %>% str_to_lower() %>% str_detect("campagnolo super record") %>% as.numeric(),
    `shimano nexus`           = `Shift Lever`     %>% str_to_lower() %>% str_detect("shimano nexus") %>% as.numeric(),
    `shimano alfine`          = `Shift Lever`     %>% str_to_lower() %>% str_detect("shimano alfine") %>% as.numeric()
  ) %>% 
  # Remove original columns  
  select(-c(`Rear Derailleur`, `Shift Lever`)) %>% 
  # Set all NAs to 0
  mutate_if(is.numeric, ~replace(., is.na(.), 0))

# 2.0 TRAINING & TEST SETS ----
bike_features_tbl <- bike_features_tbl %>% 
  
  mutate(id = row_number()) %>% 
  
  select(id, everything(), -url)

bike_features_tbl %>% distinct(category_2)

# run both following commands at the same time
set.seed(seed = 1113)
split_obj <- rsample::initial_split(bike_features_tbl, prop   = 0.80, 
                                    strata = "category_2")

# Check if testing contains all category_2 values
split_obj %>% training() %>% distinct(category_2)
split_obj %>% testing() %>% distinct(category_2)

# Assign training and test data
train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)

# We have to remove spaces and dashes from the column names
train_tbl <- train_tbl %>% set_names(str_replace_all(names(train_tbl), " |-", "_"))
test_tbl  <- test_tbl  %>% set_names(str_replace_all(names(test_tbl),  " |-", "_"))

# 3.0 LINEAR METHODS ----
# 3.1 LINEAR REGRESSION - NO ENGINEERED FEATURES ----

# 3.1.1 Model ----
?lm # from the stats package
?set_engine
?fit # then click Estimate model parameters and then fit at the bottom

model_01_linear_lm_simple <- linear_reg(mode = "regression") %>%
  set_engine("lm") %>%
  fit(price ~ category_2 + frame_material, data = train_tbl)

model_01_linear_lm_simple %>%
  predict(new_data = test_tbl)