library(h2o)
library(tidyverse)
library(rsample)
library(recipes)
library(readr)

# Inicializar H2O
h2o.init()

# Cargar el dataset
data <- read_csv("product_backorders.csv")

# Verificar la estructura de los datos
glimpse(data)

# Limpiar datos (opcional)
data_clean <- data %>%
  mutate_if(is.character, as.factor) %>%
  mutate(across(where(is.factor), as.numeric)) %>%
  replace_na(list(0))

# Dividir los datos en conjuntos de entrenamiento y prueba
set.seed(1234)
split <- initial_split(data_clean, prop = 0.85)
train <- training(split)
test <- testing(split)

# Convertir los datos a H2O Frame
train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)

# Definir la variable objetivo y las variables predictoras
y <- "went_on_backorder"  # Columna objetivo
x <- setdiff(names(train_h2o), y)

# Ejecutar AutoML con configuración ajustada
automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame = train_h2o,
  max_runtime_secs = 60, # Puede aumentar este tiempo para mejorar el resultado
  nfolds = 5,
  exclude_algos = c("DeepLearning"), # Excluir modelos que pueden fallar con más frecuencia
  seed = 1234
)

# Ver el leaderboard
leaderboard <- automl_models_h2o@leaderboard
print(leaderboard)

# Obtener el modelo líder
leader <- automl_models_h2o@leader
print(leader)

# Predecir usando el modelo líder
predictions <- h2o.predict(leader, test_h2o)
print(predictions)

# Guardar el modelo líder
model_path <- h2o.saveModel(leader, path = "04_Modeling/h20_models_back/", force = TRUE)
print(model_path)

# Cargar el modelo guardado (si es necesario)
loaded_model <- h2o.loadModel(model_path)
print(loaded_model)

# Predecir usando el modelo cargado (para verificar)
loaded_predictions <- h2o.predict(loaded_model, test_h2o)
print(loaded_predictions)
