# Función personalizada para plot_features
plot_features_custom <- function(explanation) {
    # Extraer la predicción como valor numérico
    prediction_value <- as.numeric(explanation$prediction[[1]]$No)
    
    ggplot(explanation, aes(x = reorder(feature, feature_weight), y = feature_weight, fill = feature_weight > 0)) +
        geom_col() +
        coord_flip() +
        theme_minimal() +
        labs(
            title = paste("Case:", unique(explanation$case)),
            subtitle = paste("Label:", unique(explanation$label), 
                             "\nProbability:", round(prediction_value, 2), 
                             "\nExplanation Fit:", round(as.numeric(unique(explanation$model_r2)), 2)),
            x = "Feature",
            y = "Weight"
        ) +
        scale_fill_manual(values = c("red", "blue"), 
                          labels = c("Contradicts", "Supports")) +
        theme(legend.position = "bottom", 
              legend.title = element_blank())
}

# Graficar las características para el primer caso
plot_features_custom(case_1)

