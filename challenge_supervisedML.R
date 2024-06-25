# Convertir las columnas a factores y alinear los niveles
convert_to_factors <- function(data, train_levels) {
  for (col in names(train_levels)) {
    if (col %in% names(data)) {
      data[[col]] <- factor(data[[col]], levels = train_levels[[col]])
    }
  }
  return(data)
}

# Guardar los niveles de los factores del conjunto de entrenamiento
train_levels <- lapply(train_tbl, levels)

# Convertir los datos de prueba para que coincidan con los niveles de entrenamiento
test_transformed_tbl <- convert_to_factors(test_transformed_tbl, train_levels)

# Convertir los datos de entrenamiento para que coincidan con los niveles de entrenamiento (opcional)
train_transformed_tbl <- convert_to_factors(train_tbl, train_levels)

# Definir la receta ajustada
recipe_obj <- recipe(price ~ ., data = train_tbl) %>%
  step_rm(category_1, category_3, gender) %>%  # Ajustar esto según las columnas disponibles
  step_other(all_nominal(), threshold = 0.01) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  prep()

# Transformar los datos de entrenamiento y prueba
train_transformed_tbl <- bake(recipe_obj, new_data = train_tbl)
test_transformed_tbl <- bake(recipe_obj, new_data = test_tbl)

# Asegurarnos de que las columnas necesarias estén presentes
required_columns <- c("model", "frame_material", "category_2", "price", "weight", "category_1", "category_3", "gender", "url")
missing_columns <- setdiff(required_columns, colnames(test_transformed_tbl))

# Añadir columnas faltantes si es necesario
if (length(missing_columns) > 0) {
  for (col in missing_columns) {
    test_transformed_tbl[[col]] <- test_tbl[[col]]
  }
}

# Calcular las métricas nuevamente
metrics <- calc_metrics(model_fit, new_data = test_transformed_tbl)
print(metrics)
