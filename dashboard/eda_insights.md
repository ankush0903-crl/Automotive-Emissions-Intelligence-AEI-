# Exploratory Data Analysis (EDA) Insights

## Distributions
The CO2 Emissions distribution is roughly normal (bell-shaped) after outlier removal, meaning most vehicles fall in the average range, with fewer extremely clean or heavily polluting vehicles. Engine size shows multiple peaks corresponding to common engine sizes (like 2.0L and 3.5L).

## Correlations
The factor that affects CO2 the most positively is **Fuel Consumption City (L/100 km)** (correlation: 0.89).
The factor that affects CO2 the most negatively is **Fuel Consumption Comb (mpg)** (correlation: -0.90).
Highly positively correlated features like Engine Size and Fuel Consumption Comb directly drive CO2 emissions up, whereas higher MPG heavily reduces CO2.

## Feature Importance
According to the Random Forest model, the most impactful feature for predicting CO2 emissions is **Fuel Consumption Comb (L/100 km)**, followed by **Fuel Consumption Comb (mpg)**.
Engine size and fuel consumption metrics play the overwhelmingly largest role compared to categorical variables like Make or Vehicle Class.
