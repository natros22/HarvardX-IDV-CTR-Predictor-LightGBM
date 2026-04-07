
# IDV: CTR Predictive Modeling on High-Cardinality Data using LightGBM
# HarvardX Data Science Professional Certificate: PH125.9x 
# author: Fidan Alasgarova
# date: 07 April 2026



###############################################################################
#  Load subset of Avazu dataset from the Github repository 
###############################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(recosystem)) install.packages("recosystem")
if(!require(lightgbm)) install.packages("lightgbm")
if(!require(ROCR)) install.packages("ROCR")

library(dplyr)
library(ggplot2)
library(tidyverse)
library(recosystem)
library(readr)

data_url <- "https://raw.githubusercontent.com/natros22/HarvardX-IDV-CTR-Predictor-Avazu/main/avazu_data.csv.gz"

local_file <- "avazu_dataset.csv.gz"

options(timeout = 600)

download.file(data_url, local_file, mode = "wb")

set.seed(123)

avazu_subset <- read_csv(local_file, show_col_types = FALSE)

# Dimenstions
dim(avazu_subset)

# Column Names
colnames(avazu_subset)

# Number of Duplicates
sum(duplicated(avazu_subset))

################################################################################
## Sparsity analysis

# Total impressions
total_impressions <- nrow(avazu_subset)

# Count the non-clicks
non_clicks <- sum(avazu_subset$click == 0)

# Sparsity
sparsity <- non_clicks / total_impressions

sparsity 

################################################################################

# Number of Clicks
sum(avazu_subset$click)

# Modified Hour Feature
avazu_subset <- avazu_subset %>%
  mutate(
    hour_num = as.integer(substr(hour, 7, 8)))
head(avazu_subset)

# Impressions per Hour
table(substr(as.character(avazu_subset$hour), 7, 8))

# CTR by Hour
avazu_subset %>%
  group_by(hour_num) %>%
  summarise(ctr = mean(click)) %>%
  ggplot(aes(x = hour_num, y = ctr)) +
  geom_col(fill = "#5E9BD8") +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "CTR by Hour", x = "Hour of Day", y = "CTR")

# Number or ad IDs 
length(unique(avazu_subset$C14))

# CTR by ad ID
c14_ctr_analysis <- avazu_subset %>%
  group_by(C14) %>%
  summarise(
    clicks = sum(click),
    impressions = n(),
    ctr = sum(click) / n())
c14_ctr_analysis

# Aggregating data by C14
c14_summary <- avazu_subset %>%
  group_by(C14) %>%
  summarise(
    total_impressions = n(),
    total_clicks = sum(click),
    ctr = total_clicks / total_impressions) %>%
  filter(total_impressions > 100)
ggplot(c14_summary, aes(x = total_impressions, y = total_clicks, color = ctr)) +
  geom_point(alpha = 0.8, size = 2) +
  scale_color_gradient(low = "navy", high = "yellow") +
  labs(title = "Ad Performance by C14: Clicks vs. Impressions",
    x = "Total Impressions",
    y = "Total Clicks",
    color = "CTR")

# Number of Group IDs
length(unique(avazu_subset$C17))

# CTR of top 15 Ad Groups
c17_summary <- avazu_subset %>%
  group_by(C17) %>%
  summarise(
    impressions = n(),
    ctr = mean(click),
    unique_ads = n_distinct(C14)) %>%
  arrange(desc(impressions))
c17_summary %>%
  top_n(15, impressions) %>%
  ggplot(aes(x = reorder(as.factor(C17), ctr), y = ctr)) +
  geom_bar(stat = "identity", fill = "#007BC0") +
  coord_flip() +
  labs(title = "CTR of Top 15 Ad Groups (C17)", x = "Ad Group ID", y = "CTR")

# Distribution of Clicks by placement ID
avazu_placement <- avazu_subset %>%
  mutate(placement_id = ifelse(site_id == "85f751fd", app_id, site_id))
avazu_placement %>%
  group_by(placement_id) %>%
  summarise(total_clicks = sum(click)) %>%
  ggplot(aes(x = total_clicks)) +
  geom_histogram(bins = 15, fill = "gold", color = "white") +
  scale_x_log10() +
  labs(title = "Distribution of Clicks by Placement",
       x = "Total Clicks",
       y = "Frequency of Placements")

# Placement by Clicks VS CTR
placement_performance <- avazu_placement %>%
  group_by(placement_id) %>%
  summarise(
    total_clicks = sum(click),
    total_impressions = n(),
    ctr = total_clicks / total_impressions) %>%
  filter(total_clicks > 10) 
ggplot(placement_performance, aes(x = total_clicks, y = ctr)) +
  geom_point(aes(size = total_impressions), color = "#007", alpha = 0.5) +
  scale_x_log10() +  
  labs(title = "Placement Performance: Clicks vs. CTR",
       x = "Total Clicks (Log Scale)",
       y = "Click-Through Rate (CTR)")

# Number of unique device models
length(unique(avazu_subset$device_model))

# Device Models by CTR
device_tiers <- avazu_subset %>%
  group_by(device_model) %>%
  summarise(impressions = n(), ctr = mean(click)) %>%
  mutate(ctr_tier = cut(ctr, 
                        breaks = seq(0, 1, by = 0.05), 
                        include.lowest = TRUE,
                        labels = paste0(seq(0, 95, by = 5), "-", seq(5, 100, by = 5), "%"))) %>%
  group_by(ctr_tier) %>%
  summarise(total_impressions = sum(impressions)) %>%
  filter(!is.na(ctr_tier)) 
ggplot(device_tiers, aes(x = ctr_tier, y = total_impressions, fill = total_impressions)) +
  geom_col() +
  scale_y_continuous(labels = scales::comma) +
  labs( title = "Ad Reach (Impressions) by Device Model", x = "Device Model CTR (%)", y = "Total Impressions", fill = "Volume") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Number of Device IP
ip_num <- length(unique(avazu_subset$device_ip))
ip_num

# Device IP with a Click
devices_with_clicks <- avazu_subset %>%
  filter(click == 1) %>%
  summarise(devices_with_clicks = n_distinct(device_ip))
devices_with_clicks

# Percentage of IPs with at lease one click
(devices_with_clicks/ip_num)*100

# Number of Banner Positions
sort(unique(avazu_subset$banner_pos))

# Aggregate CTR by Banner Position
banner_ctr <- avazu_subset %>%
  group_by(banner_pos) %>%
  summarise(banner_ctr = mean(click)) 
ggplot(banner_ctr, aes(x = factor(banner_pos), y = banner_ctr, fill = banner_ctr)) +
  geom_col() +
  scale_y_continuous(labels = function(x) paste0(round(x*100, 1), "%")) +
  labs(title = "CTR by Banner Position", x = "Banner Position ID", y = "Average CTR (%)", fill = "CTR") 

################################################################################
# LightGBM CTR Prediction Model
################################################################################
library(dplyr)
library(readr)
library(lightgbm)
library(ROCR)

# Ensure reproducibility
set.seed(123)

# Reload avazu_subset
avazu_subset <- read_csv(local_file, show_col_types = FALSE)

# Feature engineering
avazu_subset <- avazu_subset %>%
  mutate(
    # Hour of day
    hour_day = as.integer(substr(as.character(hour), 7, 8)),
    # User Proxy
    user_proxy = paste(device_ip, device_model, sep = "_"),
    # Placement Identity
    placement = ifelse(site_id == "85f751fd", app_id, site_id)) %>%
  # Ad appearances per hour
  group_by(hour, C14) %>%
  mutate(ad_hourly_cnt = log1p(n())) %>%
  # IP Frequency
  group_by(device_ip) %>%
  mutate(ip_cnt = log1p(n())) %>%
  ungroup()

# Define feature sets
features <- c("hour_day", "banner_pos", "placement", "device_model", 
              "user_proxy", "C14", "C17", "ip_cnt", "ad_hourly_cnt")
categorical_features <- c("banner_pos", "placement", "device_model", "user_proxy", "C14", "C17")
target <- "click"

# Train/Test split
train_idx <- sample(1:nrow(avazu_subset), 0.8 * nrow(avazu_subset))
train_set <- avazu_subset[train_idx, ]
test_set  <- avazu_subset[-train_idx, ]

# Categorical Encoding
for (col in categorical_features) {
  train_levels <- unique(train_set[[col]])
  
  # Convert to factor based on training levels only
  train_set[[col]] <- as.integer(factor(train_set[[col]], levels = train_levels)) - 1
  test_set[[col]]  <- as.integer(factor(test_set[[col]],  levels = train_levels)) - 1}

# Prepare LightGBM dataset 
dtrain <- lgb.Dataset(
  data = as.matrix(train_set %>% select(all_of(features))),
  label = train_set[[target]],
  categorical_feature = categorical_features)

dtest <- lgb.Dataset(
  data = as.matrix(test_set %>% select(all_of(features))),
  label = test_set[[target]],
  reference = dtrain)

# Optimized Parameters
params <- list(
  objective = "binary",
  metric = "binary_logloss",
  boosting = "gbdt",
  learning_rate = 0.05,    
  num_leaves = 127,        
  feature_fraction = 0.7,  
  bagging_fraction = 0.7,
  bagging_freq = 5,
  force_col_wise = TRUE,
  verbosity = -1)

# Train model
lgb_model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,          
  valids = list(test = dtest),
  early_stopping_rounds = 30)

# Predict
pred_prob <- predict(lgb_model, as.matrix(test_set %>% select(all_of(features))))
pred_obj <- prediction(pred_prob, test_set[[target]])

# Evaluate
# Calculate AUC
auc_val <- performance(pred_obj, "auc")@y.values[[1]]

# Calculate LogLoss (with epsilon to avoid Inf)
logloss_val <- -mean(test_set[[target]] * log(pred_prob + 1e-15) +
                       (1 - test_set[[target]]) * log(1 - pred_prob + 1e-15))

# Results
cat("AUC:    ", round(auc_val, 5), "\n")
cat("LogLoss:", round(logloss_val, 5), "\n")