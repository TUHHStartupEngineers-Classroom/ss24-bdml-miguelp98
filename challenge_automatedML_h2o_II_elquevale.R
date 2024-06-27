# Cargar librerías necesarias
library(h2o)
library(tidyverse)

# Cargar el dataset
data <- read.csv("product_backorders.csv")

# Convertir la columna "went_on_backorder" en un factor
data$went_on_backorder <- as.factor(data$went_on_backorder)

# Inicializar H2O
h2o.init()

# Convertir el data frame de R a un objeto H2O
data_h2o <- as.h2o(data)

# Dividir los datos en conjunto de entrenamiento y prueba
splits <- h2o.splitFrame(data_h2o, ratios = 0.8, seed = 1234)
train_h2o <- splits[[1]]
test_h2o <- splits[[2]]

# Definir las variables de predicción y objetivo
y <- "went_on_backorder"
x <- setdiff(names(train_h2o), y)


# Ejecutar H2O AutoML
automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame = train_h2o,
  max_runtime_secs = 60,
  nfolds = 5,
  seed = 1234
)

# Ver el leaderboard
lb <- automl_models_h2o@leaderboard
print(lb)


# Obtener el modelo líder
leader_model <- automl_models_h2o@leader

# Guardar el modelo líder
model_path <- h2o.saveModel(leader_model, path = "04_Modeling/h20_models_back", force = TRUE)
print(model_path)

# Cargar el modelo guardado
loaded_model <- h2o.loadModel(model_path)

# Confirmar que el modelo se ha cargado correctamente
print(loaded_model)

