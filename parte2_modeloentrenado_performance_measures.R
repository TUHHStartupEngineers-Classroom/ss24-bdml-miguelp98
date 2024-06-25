# Cargar las bibliotecas necesarias
library(h2o)
library(tidyverse)
library(cowplot)
library(glue)
library(fs)

# Inicializar H2O
h2o.init()

# Directorio donde se guardan los modelos
model_dir <- "04_Modeling/h2o_new/"

# Nombres de los modelos guardados
model_names <- c("StackedEnsemble_BestOfFamily_1_AutoML_4_20240624_171417", 
                 "StackedEnsemble_AllModels_1_AutoML_4_20240624_171417",
                 "StackedEnsemble_BestOfFamily_2_AutoML_4_20240624_171417")

# Cargar los modelos guardados
loaded_models <- lapply(model_names, function(model_name) {
  h2o.loadModel(file.path(model_dir, model_name))
})

# Ver un resumen de los datos para asegurarse de que los datos de prueba están cargados correctamente
product_backorders_tbl <- read_csv("product_backorders.csv")
glimpse(product_backorders_tbl)

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

# Crear un objeto de rendimiento para cada modelo
performances <- lapply(loaded_models, function(model) {
  h2o.performance(model, newdata = as.h2o(test_tbl))
})

# Tema personalizado
theme_new <- theme(
  legend.position  = "bottom",
  legend.key       = element_blank(),
  panel.background = element_rect(fill   = "transparent"),
  panel.border     = element_rect(color = "black", fill = NA, size = 0.5),
  panel.grid.major = element_line(color = "grey", size = 0.333)
)

# Visualización del rendimiento de los modelos
plot_h2o_performance <- function(h2o_leaderboard, newdata, order_by = c("auc", "logloss"),
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
      as_tibble() %>%
      select(threshold, tpr, fpr, precision, recall)
    
  }
  
  model_metrics_tbl <- leaderboard_tbl %>%
    mutate(metrics = map(model_id, get_model_performance_metrics, newdata_tbl)) %>%
    unnest(cols = metrics) %>%
    mutate(
      model_id = as_factor(model_id) %>% 
        # programmatically reorder factors depending on order_by
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
      as_tibble() %>%
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
  p_legend <- get_legend(p1)
  p1 <- p1 + theme(legend.position = "none")
  p <- cowplot::plot_grid(p1, p2, p3, p4, ncol = 2)
  p_title <- ggdraw() + 
    draw_label("H2O Model Metrics", size = 18, fontface = "bold", color = "#2C3E50")
  p_subtitle <- ggdraw() + 
    draw_label(glue("Ordered by {toupper(order_by)}"), size = 10, color = "#2C3E50")
  
  # Combine everything
  ret <- plot_grid(p_title, p_subtitle, p, p_legend, 
                   ncol = 1, rel_heights = c(0.05, 0.05, 1, 0.05))
  
  h2o.show_progress()
  
  return(ret)
}

# Crear el leaderboard manualmente
leaderboard <- data.frame(
  model_id = model_names,
  auc = sapply(performances, h2o.auc),
  logloss = sapply(performances, h2o.logloss)
)

# Mostrar el leaderboard
print(leaderboard)

# Usar la función plot_h2o_performance para crear las gráficas
plot_h2o_performance(leaderboard, newdata = test_tbl, order_by = "auc", size = 1.5, max_models = 3)
                 