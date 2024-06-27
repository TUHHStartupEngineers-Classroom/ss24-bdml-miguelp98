# Librerías necesarias
library(h2o)
library(tidyverse)
library(rsample)
library(recipes)
library(cowplot)
library(glue)

# Inicializar H2O
h2o.init()

# Cargar los datos
product_backorders_tbl <- read_csv("product_backorders.csv")

# Ver un resumen de los datos
glimpse(product_backorders_tbl)

# Establecer la semilla para reproducibilidad
set.seed(1113)

# Dividir los datos en conjuntos de entrenamiento y prueba
split_obj <- initial_split(product_backorders_tbl, prop = 0.85)
train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)

# Crear una receta para el preprocesamiento de los datos
recipe_obj <- recipe(went_on_backorder ~ ., data = train_tbl) %>% 
  step_zv(all_predictors()) %>% 
  step_mutate_at(national_inv, lead_time, in_transit_qty, forecast_3_month, 
                 forecast_6_month, forecast_9_month, sales_1_month, sales_3_month, 
                 sales_6_month, sales_9_month, min_bank, pieces_past_due, 
                 perf_6_month_avg, perf_12_month_avg, local_bo_qty, deck_risk, 
                 oe_constraint, ppap_risk, stop_auto_buy, rev_stop, 
                 went_on_backorder, fn = as.factor) %>% 
  prep()

# Transformar los datos usando la receta
train_tbl <- bake(recipe_obj, new_data = train_tbl)
test_tbl  <- bake(recipe_obj, new_data = test_tbl)

# Dividir los datos en un conjunto de entrenamiento y un conjunto de validación
set.seed(1234)
split_h2o <- h2o.splitFrame(as.h2o(train_tbl), ratios = c(0.85), seed = 1234)
train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]
test_h2o  <- as.h2o(test_tbl)

# Establecer la variable objetivo y los predictores
y <- "went_on_backorder"
x <- setdiff(names(train_h2o), y)

# Ejecutar H2O AutoML
automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame    = train_h2o,
  validation_frame  = valid_h2o,
  leaderboard_frame = test_h2o,
  max_runtime_secs  = 30,
  nfolds            = 5 
)

# Obtener los nombres de los mejores modelos
leaderboard <- as.data.frame(automl_models_h2o@leaderboard)
best_models <- leaderboard$model_id[1:3]

# Crear una carpeta para guardar los modelos si no existe
dir.create("04_Modeling/h2o_new", showWarnings = FALSE, recursive = TRUE)

# Guardar los mejores modelos
for (model_id in best_models) {
  model <- h2o.getModel(model_id)
  h2o.saveModel(model, path = "04_Modeling/h2o_new/")
}

# Cargar los modelos guardados
loaded_models <- lapply(list.files("04_Modeling/h2o_new/", full.names = TRUE), h2o.loadModel)

# Crear un objeto de rendimiento para cada modelo
performances <- lapply(loaded_models, function(model) h2o.performance(model, newdata = test_h2o))

# Tema personalizado
theme_new <- theme(
  legend.position  = "bottom",
  legend.key       = element_blank(),
  panel.background = element_rect(fill   = "transparent"),
  panel.border     = element_rect(color = "black", fill = NA, size = 0.5),
  panel.grid.major = element_line(color = "grey", size = 0.333)
)

# Función para visualizar el rendimiento de los modelos
plot <- plot_h2o_performance <- function(h2o_leaderboard, newdata, order_by = c("auc", "logloss"),
                                 max_models = 3, size = 1.5) {
  
  # Inputs
  leaderboard_tbl <- h2o_leaderboard %>%
    as_tibble() %>%
    slice(1:max_models)
  
  newdata_tbl <- newdata %>%
    as_tibble()
  
  # Selecting the first, if nothing is provided
  order_by      <- tolower(order_by[[1]]) 
  
  # Convert string stored in a variable to column name (symbol)
  order_by_expr <- rlang::sym(order_by)
  
  # Turn off the progress bars (opposite h2o.show_progress())
  h2o.no_progress()
  
  # 1. Model metrics
  get_model_performance_metrics <- function(model_id, test_tbl) {
    model_h2o <- h2o.getModel(model_id)
    perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))
    perf_h2o %>%
      h2o.metric() %>%
      as.tibble() %>%
      select(threshold, tpr, fpr, precision, recall)
  }
  
  model_metrics_tbl <- leaderboard_tbl %>%
    mutate(metrics = map(model_id, get_model_performance_metrics, newdata_tbl)) %>%
    unnest(cols = metrics) %>%
    mutate(
      model_id = as_factor(model_id) %>% 
        fct_reorder(!! order_by_expr, 
                    .desc = ifelse(order_by == "auc", TRUE, FALSE)),
      auc      = auc %>% 
        round(3) %>% 
        as.character() %>% 
        as_factor() %>% 
        fct_reorder(as.numeric(model_id)),
      logloss  = logloss %>% 
        round(4) %>% 
        as.character() %>% 
        as_factor() %>% 
        fct_reorder(as.numeric(model_id))
    )
  
  # 1A. ROC Plot
  p1 <- model_metrics_tbl %>%
    ggplot(aes(fpr, tpr, color = model_id, linetype = !! order_by_expr)) +
    geom_line(size = size) +
    theme_new +
    labs(title = "ROC", x = "FPR", y = "TPR") +
    theme(legend.direction = "vertical")
  
  # 1B. Precision vs Recall
  p2 <- model_metrics_tbl %>%
    ggplot(aes(recall, precision, color = model_id, linetype = !! order_by_expr)) +
    geom_line(size = size) +
    theme_new +
    labs(title = "Precision Vs Recall", x = "Recall", y = "Precision") +
    theme(legend.position = "none")
  
  # 2. Gain / Lift
  get_gain_lift <- function(model_id, test_tbl) {
    model_h2o <- h2o.getModel(model_id)
    perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl)) 
    perf_h2o %>%
      h2o.gainsLift() %>%
      as.tibble() %>%
      select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift)
  }
  
  gain_lift_tbl <- leaderboard_tbl %>%
    mutate(metrics = map(model_id, get_gain_lift, newdata_tbl)) %>%
    unnest(cols = metrics) %>%
    mutate(
      model_id = as_factor(model_id) %>% 
        fct_reorder(!! order_by_expr, 
                    .desc = ifelse(order_by == "auc", TRUE, FALSE)),
      auc  = auc %>% 
        round(3) %>% 
        as.character() %>% 
        as_factor() %>% 
        fct_reorder(as.numeric(model_id)),
      logloss = logloss %>% 
        round(4) %>% 
        as.character() %>% 
        as_factor() %>% 
        fct_reorder(as.numeric(model_id))
    ) %>%
    rename(
      gain = cumulative_capture_rate,
      lift = cumulative_lift
    )
  
  # 2A. Gain Plot
  p3 <- gain_lift_tbl %>%
    ggplot(aes(cumulative_data_fraction, gain, 
               color = model_id, linetype = !! order_by_expr)) +
    geom_line(size = size) +
    geom_segment(x = 0, y = 0, xend = 1, yend = 1, 
                 color = "red", size = size, linetype = "dotted") +
    theme_new +
    expand_limits(x = c(0, 1), y = c(0, 1)) +
    labs(title = "Gain",
         x = "Cumulative Data Fraction", y = "Gain") +
    theme(legend.position = "none")
  
  # 2B. Lift Plot
  p4 <- gain_lift_tbl %>%
    ggplot(aes(cumulative_data_fraction, lift, 
               color = model_id, linetype = !! order_by_expr)) +
    geom_line(size = size) +
    geom_segment(x = 0, y = 1, xend = 1, yend = 1,
                 color = "red", size = size, linetype = "dotted") +
    theme_new +
    expand_limits(x = c(0, 1), y = c(0, 1)) +
    labs(title = "Lift",
         x = "Cumulative Data Fraction", y = "Lift") +
    theme(legend.position = "none") 
  
  
  # Combine using cowplot
  
  # cowplot::get_legend extracts a legend from a ggplot object
  p_legend <- get_legend(p1)
  # Remove legend from p1
  p1 <- p1 + theme(legend.position = "none")
  
  # cowplot::plot_grid() combines multiple ggplots into a single cowplot object
  p <- cowplot::plot_grid(p1, p2, p3, p4, ncol = 2)
  
  # cowplot::ggdraw() sets up a drawing layer
  p_title <- ggdraw() + 
    
    # cowplot::draw_label() draws text on a ggdraw layer / ggplot object
    draw_label("H2O Model Metrics", size = 18, fontface = "bold", 
               color = "#2C3E50")
  
  p_subtitle <- ggdraw() + 
    draw_label(glue("Ordered by {toupper(order_by)}"), size = 10,  
               color = "#2C3E50")
  
  # Combine everything
  ret <- plot_grid(p_title, p_subtitle, p, p_legend, 
                   
                   # Adjust the relative spacing, so that the legends always fit
                   ncol = 1, rel_heights = c(0.05, 0.05, 1, 0.05 * max_models))
  
  h2o.show_progress()
  
  return(ret)
  
}

# Visualización del rendimiento de los modelos
automl_models_h2o@leaderboard %>%
  plot_h2o_performance(newdata = test_tbl, order_by = "logloss", 
                       size = 0.5, max_models = 4)

print(plot)
                 