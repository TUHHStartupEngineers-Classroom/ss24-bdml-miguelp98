# Cargar librerías necesarias
library(tidymodels)
library(recipes)
library(yardstick)

# Verificar columnas en train_tbl
colnames(train_tbl)

# Definir la receta ajustada sin eliminar columnas necesarias
recipe_obj <- recipe(price ~ ., data = train_tbl) %>%
  step_rm(category_1, category_3, gender) %>%  # Ajustar esto según las columnas disponibles
  step_dummy(all_nominal(), one_hot = TRUE)

# Crear el modelo
model <- linear_reg() %>% 
  set_engine("lm")

# Crear el flujo de trabajo y agregar la receta y el modelo
workflow_obj <- workflow() %>% 
  add_recipe(recipe_obj) %>% 
  add_model(model)

# Entrenar el modelo con el flujo de trabajo
model_fit <- workflow_obj %>% 
  fit(data = train_tbl)

# Definir la función de cálculo de métricas
calc_metrics <- function(model_fit, new_data) {
  predictions <- predict(model_fit, new_data = new_data) %>% 
    bind_cols(new_data %>% select(price))
  
  predictions %>% 
    yardstick::metrics(truth = price, estimate = .pred)
}

# Transformar los datos de prueba usando la receta
test_transformed_tbl <- bake(prep(recipe_obj), new_data = test_tbl)

# Asegurarnos de que las columnas necesarias estén presentes
required_columns <- c("model", "frame_material", "category_2", "price", "weight", "category_1", "category_3", "gender", "url")
missing_columns <- setdiff(required_columns, colnames(test_transformed_tbl))

# Verificar columnas faltantes y añadirlas si es necesario
if (length(missing_columns) > 0) {
  for (col in missing_columns) {
    test_transformed_tbl[[col]] <- test_tbl[[col]]
  }
}

# Calcular las métricas
metrics <- calc_metrics(model_fit, new_data = test_transformed_tbl)
print(metrics)
