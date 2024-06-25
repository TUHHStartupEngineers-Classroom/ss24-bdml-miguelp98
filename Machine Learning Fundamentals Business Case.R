library(readxl)
library(tidyverse)
library(broom)
library(umap)
library(ggrepel) # Addon for ggplot, so that the labels do not overlap

stores_tbl      <- read_excel(path  = "breakfast_at_the_frat.xlsx", 
                              sheet = "dh Store Lookup",
                              skip  = 1)
products_tbl    <- read_excel(path  = "breakfast_at_the_frat.xlsx", 
                              sheet = "dh Products Lookup",
                              skip  = 1)
transaction_tbl <- read_excel(path  = "breakfast_at_the_frat.xlsx",
                              sheet = "dh Transaction Data", 
                              skip  = 1)

orderlines_tbl <- transaction_tbl %>% 
  left_join(products_tbl) %>% 
  left_join(stores_tbl, by = c("STORE_NUM" = "STORE_ID"))


# 1.1 Get Customer Trends ----
customer_trends_tbl <- orderlines_tbl %>%
  
  mutate(BRANDED = ifelse(MANUFACTURER == "PRIVATE LABEL", "no", "yes")) %>% 
  select(STORE_NAME, PRICE, UPC, DESCRIPTION, CATEGORY, SUB_CATEGORY, BRANDED, UNITS) %>%
  
  # Summarization and group by
  group_by(STORE_NAME, PRICE, UPC, DESCRIPTION, CATEGORY, SUB_CATEGORY, BRANDED) %>%
  summarise(QUANTITY_PURCHASED = sum(UNITS)) %>%
  ungroup() %>%
  
  # Proportion
  group_by(STORE_NAME) %>%
  mutate(PROP_OF_TOTAL = QUANTITY_PURCHASED / sum(QUANTITY_PURCHASED)) %>%
  ungroup()

# 1.2 Convert to User-Item Format (e.g. Customer-Product) ----

customer_product_tbl <- customer_trends_tbl %>%
  
  select(STORE_NAME, UPC, PROP_OF_TOTAL) %>%
  pivot_wider(names_from = UPC, values_from = PROP_OF_TOTAL, values_fill = 0) %>%
  ungroup()

# 2.0 MODELING: K-MEANS CLUSTERING ----

# 2.1 Performing K-Means ----

kmeans_obj <- customer_product_tbl %>%
  select(-STORE_NAME) %>%
  kmeans(centers = 3, nstart = 100)

# The group that each product is assigned to for each product   
kmeans_obj$cluster

# 2.2 Tidying a K-Means Object ----
# return the centers information for the kmeans model
broom::tidy(kmeans_obj) %>% glimpse()

# return the overall summary metrics for the model
# Including the tot.withinss for the skree plot
broom::glance(kmeans_obj)

# Add the clusters to the data
broom::augment(kmeans_obj, customer_product_tbl) %>%
  select(STORE_NAME, .cluster)

# 2.3 How many centers (customer groups) to use? ----

# Functions that works on 1 element 
# Setting default centers to 3

kmeans_mapper <- function(centers = 3) {
  
  customer_product_tbl %>%
    select(-STORE_NAME) %>%
    kmeans(centers = centers, nstart = 100)
}

3 %>% kmeans_mapper() %>% glance()


# Mapping the function to many elements
kmeans_mapped_tbl <- tibble(centers = 1:15) %>%
    mutate(k_means = centers %>% map(kmeans_mapper)) %>%
    mutate(glance  = k_means %>% map(glance))

kmeans_mapped_tbl %>%
  unnest(glance) %>%
  select(centers, tot.withinss)


# 2.4 Skree Plot ----

kmeans_mapped_tbl %>%
  unnest(glance) %>%
  select(centers, tot.withinss) %>%
  
  # Visualization
  ggplot(aes(centers, tot.withinss)) +
  geom_point(color = "#2DC6D6", size = 4) +
  geom_line(color = "#2DC6D6", size = 1) +
  # Add labels (which are repelled a little)
  ggrepel::geom_label_repel(aes(label = centers), color = "#2DC6D6") + 
  
  # Formatting
  labs(title = "Skree Plot",
       subtitle = "Measures the distance each of the customer are from the closes K-Means center",
       caption = "Conclusion: Based on the Scree Plot, we select 3 clusters to segment the customer base.")

# 3.0 VISUALIZATION: UMAP ----

# 3.1 Use UMAP to get 2-D Projection ----
umap_obj <- customer_product_tbl %>%
  select(-STORE_NAME) %>%
  umap()

umap_results_tbl <- umap_obj$layout %>%
  as_tibble(.name_repair = "unique") %>% # argument is required to set names in the next step
  set_names(c("x", "y")) %>%
  bind_cols(
    customer_product_tbl %>% select(STORE_NAME)
  )

umap_results_tbl %>%
  ggplot(aes(x, y)) +
  geom_point() + 
  geom_label_repel(aes(label = STORE_NAME), size = 3, max.overlaps = 100)

# 3.2 Use K-Means to Add Cluster Assignments ----
umap_results_tbl

# Get the data for the third element (which we have chosen in the skree plot)
kmeans_3_obj <- kmeans_mapped_tbl %>%
  pull(k_means) %>%
  pluck(3)

# Convert it to a tibble with broom
kmeans_3_clusters_tbl <- kmeans_3_obj %>% 
  augment(customer_product_tbl) %>%
  # Select the data we need
  select(STORE_NAME, .cluster)

# Bind data together
umap_kmeans_3_results_tbl <- umap_results_tbl %>%
  left_join(kmeans_3_clusters_tbl)

# 3.3 Visualize UMAP'ed Projections with Cluster Assignments ----

umap_kmeans_3_results_tbl %>%
  mutate(label_text = str_glue("Customer: {STORE_NAME}
                                 Cluster: {.cluster}")) %>%
  
  ggplot(aes(x, y, color = .cluster)) +
  
  # Geometries
  geom_point() +
  geom_label_repel(aes(label = label_text), size = 2, fill = "#282A36", max.overlaps = 100) +
  
  # Formatting
  scale_color_manual(values=c("#2d72d6", "#2dc6d6", "#2dd692")) +
  labs(title = "Customer Segmentation: 2D Projection",
       subtitle = "UMAP 2D Projection with K-Means Cluster Assignment",
       caption = "Conclusion: 3 Customer Segments identified using 2 algorithms") +
  theme(legend.position = "none")

# 4.0 ANALYZE PURCHASING TRENDS ----

cluster_trends_tbl <- customer_trends_tbl %>%
  left_join(umap_kmeans_3_results_tbl, by = "STORE_NAME") %>% # Asegúrate de que "STORE_NAME" esté en ambas tablas
  # 4.2 Clasificación del precio
  mutate(price_bin = case_when(
    PRICE <= quantile(PRICE, probs = 0.33) ~ "low",
    PRICE <= quantile(PRICE, probs = 0.66) ~ "medium",
    TRUE ~ "high"
  )) %>%
  # 4.3 Reorganización de columnas
  select(.cluster, UPC, DESCRIPTION, PRICE, CATEGORY, QUANTITY_PURCHASED) %>%
  # 4.4 Agregación con group_by_at()
  group_by_at(vars(.cluster, BRANDED)) %>%
  summarise(TOTAL_QUANTITY = sum(QUANTITY_PURCHASED)) %>%
  ungroup() %>%
  # 4.5 Normalización
  group_by(.cluster) %>%
  mutate(PROP_OF_TOTAL = TOTAL_QUANTITY / sum(TOTAL_QUANTITY)) %>%
  ungroup()

get_cluster_trends <- function(cluster = 1) {
  cluster_trends_tbl %>%
    filter(.cluster == cluster) %>%
    arrange(desc(PROP_OF_TOTAL)) %>%
    mutate(cum_prop = cumsum(PROP_OF_TOTAL)) %>%
    View()
}

get_cluster_trends(cluster = 1)
get_cluster_trends(cluster = 2)
get_cluster_trends(cluster = 3)