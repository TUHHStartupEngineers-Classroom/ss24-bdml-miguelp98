# Cargar las librerías necesarias
library(tidyverse)
library(readxl)
library(rsample)
library(recipes)
library(h2o)

# Función para procesar datos de HR
process_hr_data_readable <- function(data, definitions_tbl) {
  definitions_list <- definitions_tbl %>%
    fill(...1, .direction = "down") %>%
    filter(!is.na(...2)) %>%
    separate(...2, into = c("key", "value"), sep = " '", remove = TRUE) %>%
    rename(column_name = ...1) %>%
    mutate(key = as.numeric(key)) %>%
    mutate(value = value %>% str_replace(pattern = "'", replacement = "")) %>%
    split(.$column_name) %>%
    map(~ select(., -column_name)) %>%
    map(~ mutate(., value = as_factor(value))) 
  
  for (i in seq_along(definitions_list)) {
    list_name <- names(definitions_list)[i]
    colnames(definitions_list[[i]]) <- c(list_name, paste0(list_name, "_value"))
  }
  
  data_merged_tbl <- list(HR_Data = data) %>%
    append(definitions_list, after = 1) %>%
    reduce(left_join) %>%
    select(-one_of(names(definitions_list))) %>%
    set_names(str_replace_all(names(.), pattern = "_value", replacement = "")) %>%
    select(sort(names(.))) %>%
    mutate_if(is.character, as.factor) %>%
    mutate(
      BusinessTravel = BusinessTravel %>% fct_relevel("Non-Travel", "Travel_Rarely", "Travel_Frequently"),
      MaritalStatus  = MaritalStatus %>% fct_relevel("Single", "Married", "Divorced")
    )
  
  return(data_merged_tbl)
}

# Cargar los datos
employee_attrition_tbl <- read_csv("datasets-1067-1925-WA_Fn-UseC_-HR-Employee-Attrition.csv")
definitions_raw_tbl    <- read_excel("data_definitions.xlsx", sheet = 1, col_names = FALSE)

# Procesar los datos
employee_attrition_readable_tbl <- process_hr_data_readable(employee_attrition_tbl, definitions_raw_tbl)

# Dividir los datos en entrenamiento y prueba
set.seed(1113)
split_obj <- rsample::initial_split(employee_attrition_readable_tbl, prop = 0.85)
train_readable_tbl <- training(split_obj)
test_readable_tbl  <- testing(split_obj)

# Preparar la receta
recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
  step_zv(all_predictors()) %>%
  step_mutate_at(c("JobLevel", "StockOptionLevel"), fn = as.factor) %>%
  prep()

# Transformar los datos según la receta
train_tbl <- bake(recipe_obj, new_data = train_readable_tbl)
test_tbl  <- bake(recipe_obj, new_data = test_readable_tbl)

# Inicializar H2O
h2o.init()

# Convertir los datos a formato H2O
train_h2o <- as.h2o(train_tbl)
test_h2o  <- as.h2o(test_tbl)

# Ejecutar AutoML
automl_models_h2o <- h2o.automl(
  x = setdiff(names(train_h2o), "Attrition"),
  y = "Attrition",
  training_frame = train_h2o,
  max_runtime_secs = 60,
  nfolds = 5,
  seed = 1234
)

# Obtener el modelo líder
leader_model <- automl_models_h2o@leader

# Guardar el modelo líder
model_path <- h2o.saveModel(leader_model, path = "04_Modeling/h20_dataset", force = TRUE)
print(model_path)

# Cargar el modelo guardado
loaded_model <- h2o.loadModel(model_path)

# Confirmar que el modelo se ha cargado correctamente
print(loaded_model)

# Hacer predicciones con el modelo cargado
predictions <- h2o.predict(loaded_model, newdata = test_h2o)
print(predictions)
