# Cargar librerías necesarias
library(tidyverse)
library(GGally)

# Cargar datos
employee_attrition_tbl <- read_csv("datasets-1067-1925-WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Definir la función plot_ggpairs
plot_ggpairs <- function(data, color = NULL, density_alpha = 0.5) {
  color_expr <- enquo(color)
  
  if (rlang::quo_is_null(color_expr)) {
    g <- data %>%
      ggpairs(lower = "blank") 
  } else {
    color_name <- quo_name(color_expr)
    
    g <- data %>%
      ggpairs(mapping = aes_string(color = color_name), 
              lower = "blank", legend = 1,
              diag = list(continuous = wrap("densityDiag", alpha = density_alpha))) +
      theme(legend.position = "bottom")
  }
  
  return(g)
}

employee_attrition_tbl %>%
  select(Attrition, MonthlyIncome) %>%
  plot_ggpairs(Attrition)

employee_attrition_tbl %>%
  select(Attrition, PercentSalaryHike) %>%
  plot_ggpairs(Attrition)

employee_attrition_tbl %>%
  select(Attrition, StockOptionLevel) %>%
  plot_ggpairs(Attrition)


employee_attrition_tbl %>%
  select(Attrition, EnvironmentSatisfaction) %>%
  plot_ggpairs(Attrition)

employee_attrition_tbl %>%
  select(Attrition, WorkLifeBalance) %>%
  plot_ggpairs(Attrition)

employee_attrition_tbl %>%
  select(Attrition, JobInvolvement) %>%
  plot_ggpairs(Attrition)

employee_attrition_tbl %>%
  select(Attrition, OverTime) %>%
  plot_ggpairs(Attrition)


employee_attrition_tbl %>%
  select(Attrition, TrainingTimesLastYear) %>%
  plot_ggpairs(Attrition)

employee_attrition_tbl %>%
  select(Attrition, YearsAtCompany) %>%
  plot_ggpairs(Attrition)

employee_attrition_tbl %>%
  select(Attrition, YearsSinceLastPromotion) %>%
  plot_ggpairs(Attrition)





