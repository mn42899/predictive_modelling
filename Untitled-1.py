# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS, GMM
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score

# %%
df = pd.read_csv('https://raw.githubusercontent.com/mn42899/predictive_modelling/refs/heads/main/data_set_hackathon_2024.csv')
df

# %% [markdown]
# Hackathon Questions to Answer:
# 
# 1. How many distinct orders are received in each month?
# 2. Which items are demanded (i.e., classification-choice model)?
# 3. What is the quantity demanded for each item in these orders?
# 4. What is the demand lead time of these orders (i.e., time elapsed from the instant when
# an order is received until its delivery)?

# %% [markdown]
# Prof Notes:
# 
# You may notice that dataset is transactional. First, you must group the transactions to have monthly information—that is, how many distinct orders are received each month (i.e., counting unique order codes grouped by the month of the order date)? Then, you can develop a predictive model (e.g., time series) to forecast the number of distinct orders.   
#  
# Second, you must develop a classification model (choice model) for the customers’ selection of the products. Here, please be careful. Given the seasonality in the apparel industry, the season must be added as an explanatory variable to the choice model. Season is a categorical variable, so you must use get-dummies or an appropriate encoding scheme.
#  
# The third uncertainty is quantity demanded. You may use empirical data to characterize this uncertainty. For example, empirical quantile for each product can be used like statistical distribution of quantity demanded for them ( https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html).
#  
# You can follow a similar approach to analyze the demand lead time. Empirical quantile information can also be useful for the demand lead time.
#  
# After you characterize each uncertainty by predictive models, classification methods, or empirically, you can apply Monte Carlo simulation to consolidate them into total demand. After simulation, you must analyze simulated data and distinguish advance demand from urgent demand. At this stage, you can compare the simulated advance demand with the actual advance demand in the test set. This comparison helps you calibrate the data and improve the forecast accuracy.  

# %%
df.describe()

# %%
df.info()

# %%
# Convert date columns to datetime
df['order_date'] = pd.to_datetime(df['order_date'], format='%d.%m.%Y')
df['requested_delivery_date'] = pd.to_datetime(df['requested_delivery_date'], format='%d.%m.%Y')

# Verify the conversion
print(df[['order_date', 'requested_delivery_date']].dtypes)

# %%
# convert items to numeric
df['items'] = pd.to_numeric(df['items'], errors='coerce')

# %%
df.isnull().sum()

# %% [markdown]
# # EDA

# %%
# Count distinct customer order codes
distinct_customer_order_codes = df['Customer Order Code'].nunique()

print(f"Distinct Customer Order Codes: {distinct_customer_order_codes}")

# %%
# Group by 'Customer Order Code' and check the number of unique currencies
curr_check = df.groupby('Customer Order Code')['Curr'].nunique()

# Add a column to flag if the currency is inconsistent (more than 1 unique currency)
curr_check_df = curr_check.reset_index()
curr_check_df.rename(columns={'Curr': 'Unique Currencies'}, inplace=True)
curr_check_df['Is_Inconsistent'] = curr_check_df['Unique Currencies'] > 1

# Display the results
print(curr_check_df)

# %%
# checking if any order codes have different currencies from each other - none do!
inconsistent_orders = curr_check_df[curr_check_df['Is_Inconsistent']]
print(inconsistent_orders)

# %% [markdown]
# ### Calculating Demand Lead Time

# %%
# First have to calculate the demand lead time

# Demand Lead Time = Requested Delivery Date - Order Date
# Gonna need EDA and feature engineering to create summary statistics (nedian + mean) for demand lead times by region, product, or customer
# Can also segment lead times into categories (e.g. short, medium, long)

df['Demand_lead_time'] = df['requested_delivery_date'] - df['order_date']
df

# %%
# just keeping this here so we can look at individual customer codes to see if anything is weird
filtered_df = df[df['Customer Order Code'] == 3201061588]
filtered_df

# %% [markdown]
# Im going to have to change the demand lead time... I think it has to be converted using quantile metrics

# %%
# Convert 'Demand_lead_time' to numeric in days
df['Demand_lead_time'] = pd.to_timedelta(df['Demand_lead_time'], errors='coerce').dt.days

# Replace negative or invalid lead times with 0
df['Demand_lead_time'] = df['Demand_lead_time'].apply(lambda x: max(x, 0) if pd.notnull(x) else 0)

# %% [markdown]
# ### Creating a separate df that merges all of the same customer codes together

# %%
import pandas as pd

# Group by 'Customer Order Code' and aggregate the other columns
aggregated_df = df.groupby('Customer Order Code').agg({
    'order_date': 'min',  # Earliest order date
    'requested_delivery_date': 'max',  # Latest requested delivery date
    'Customer Country Code': 'first',  # Use the first occurrence
    'Product Code': ', '.join,  # Combine all product codes
    'Description': ', '.join,  # Combine all descriptions
    'order_type': 'first',  # Use the first occurrence
    'value': 'sum',  # Sum of order values
    'Curr': 'first',  # Use the first occurrence
    'items': 'sum',  # Total items
    'Route': 'first',  # Use the first occurrence
    'Demand_lead_time': 'mean'  # Average of cleaned lead times
}).reset_index()

# Display the result
aggregated_df

# %%
aggregated_df['Curr'].value_counts()

# %%
# Define exchange rates relative to EUR
exchange_rates_to_eur = {
    'EUR': 1,     # EUR is already the base currency
    'RUB': 0.012, # Example rate: 1 RUB = 0.012 EUR
    'CHF': 0.95,  # Example rate: 1 CHF = 0.95 EUR
    'CZK': 0.04,  # Example rate: 1 CZK = 0.04 EUR
    'PLN': 0.21,  # Example rate: 1 PLN = 0.21 EUR
    'NOK': 0.09,  # Example rate: 1 NOK = 0.09 EUR
    'DKK': 0.13,  # Example rate: 1 DKK = 0.13 EUR
    'SEK': 0.088, # Example rate: 1 SEK = 0.088 EUR
    'GBP': 1.17   # Example rate: 1 GBP = 1.17 EUR
}

# Add a new column with values converted to EUR
aggregated_df['value_in_eur'] = aggregated_df.apply(
    lambda row: row['value'] * exchange_rates_to_eur.get(row['Curr'], 1), axis=1
)

# Round the 'value_in_eur' column to two decimal places
aggregated_df['value_in_eur'] = aggregated_df['value_in_eur'].round(2)

# Display the updated DataFrame
aggregated_df

# %%
# Order histogram by month-year
import pandas as pd
import matplotlib.pyplot as plt

# Ensure 'order_date' is in datetime format
aggregated_df['order_date'] = pd.to_datetime(aggregated_df['order_date'], errors='coerce')

# Extract month-year from the 'order_date'
aggregated_df['month_year'] = aggregated_df['order_date'].dt.to_period('M')

# Count the number of orders for each month-year
orders_by_month_year = aggregated_df['month_year'].value_counts().sort_index()

# Plot the histogram
plt.figure(figsize=(12, 6))
orders_by_month_year.plot(kind='bar')
plt.title('Histogram of Orders by Month-Year', fontsize=14)
plt.xlabel('Month-Year', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# line plot & rolling average

# Aggregate orders by month-year
time_series = aggregated_df.groupby('month_year')['value_in_eur'].sum()

# Plot total order value
plt.figure(figsize=(12, 6))
plt.plot(time_series.index.astype(str), time_series.values, label='Total Order Value')
plt.title('Total Order Value Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Value (in EUR)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Box Plot for demand lead time
plt.figure(figsize=(8, 6))
sns.boxplot(x=aggregated_df['Demand_lead_time'])
plt.title('Box Plot of Demand Lead Time')
plt.xlabel('Lead Time (days)')
plt.show()

# %%
# Bar Plot for orders by Customer Country Code
plt.figure(figsize=(10, 6))
aggregated_df['Customer Country Code'].value_counts().plot(kind='bar')
plt.title('Number of Orders by Customer Country Code')
plt.xlabel('Country Code')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.show()

# Pie Chart for percentage of orders by Route
aggregated_df['Route'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8))
plt.title('Percentage of Orders by Route')
plt.ylabel('')
plt.show()

# %%
# Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(aggregated_df[['value_in_eur', 'items', 'Demand_lead_time']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot for value_in_eur vs. items
plt.figure(figsize=(8, 6))
sns.scatterplot(data=aggregated_df, x='items', y='value_in_eur')
plt.title('Scatter Plot: Items vs. Order Value')
plt.xlabel('Number of Items')
plt.ylabel('Value (in EUR)')
plt.show()

# %%
# Bar Plot for average demand lead time by Route
avg_lead_time_by_route = aggregated_df.groupby('Route')['Demand_lead_time'].mean()
avg_lead_time_by_route.plot(kind='bar', figsize=(10, 6))
plt.title('Average Demand Lead Time by Route')
plt.xlabel('Route')
plt.ylabel('Lead Time (days)')
plt.tight_layout()
plt.show()

# %%
# Stacked bar chart for order type over time
order_type_by_time = aggregated_df.groupby(['month_year', 'order_type'])['value_in_eur'].sum().unstack()
order_type_by_time.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Order Type Distribution Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Order Value (EUR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Area Plot for cumulative revenue over time
cumulative_revenue = time_series.cumsum()
plt.figure(figsize=(12, 6))
plt.fill_between(cumulative_revenue.index.astype(str), cumulative_revenue.values, alpha=0.5)
plt.title('Cumulative Revenue Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Cumulative Revenue (EUR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# # How many distinct orders are received in each month?
# 

# %% [markdown]
# Based on the histogram, this is how many distinct orders are received in each month with most being made in the month of July. As for the forecasting, it is listed as the 

# %%
# distinct orders by month
import pandas as pd
import matplotlib.pyplot as plt

# Ensure 'order_date' is in datetime format
aggregated_df['order_date'] = pd.to_datetime(aggregated_df['order_date'], errors='coerce')

# Extract the month (as the name of the month)
aggregated_df['month'] = aggregated_df['order_date'].dt.month_name()

# Count orders by month
orders_by_month = aggregated_df['month'].value_counts().reindex(
    ['January', 'February', 'March', 'April', 'May', 'June', 
     'July', 'August', 'September', 'October', 'November', 'December']
)

# Plot the order count by month
plt.figure(figsize=(10, 6))
orders_by_month.plot(kind='bar', color='skyblue')
plt.title('Order Count by Month')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# # Feature Engineering

# %%
#Creating a season categorical variable to be added to our explanatory variables
df['month'] = df['order_date'].dt.month

#Mapping the months to the seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

#Creating new categorical variable 
df['season'] = df['month'].apply(get_season)

df.head()

# %%
#Creating monthly orders - Aggregating orders by months

df['order_month'] = df['order_date'].dt.to_period('M')

monthly_orders = df.groupby('order_month').size()

# Ensure the index is in Timestamp format
monthly_orders.index = monthly_orders.index.to_timestamp()

 #Plot historical data
plt.figure(figsize=(10, 6))
plt.plot(monthly_orders, label='Historical Data', color='blue', marker='o')

# %%
#Only 27 observations.....

monthly_orders.reset_index()

# %% [markdown]
# ### ARIMA Model Implementation

# %%
# Data Preparation - Creating a time series where the index is the month and the value is the number of ordered 

# From the results, the data is stationary
result = adfuller(monthly_orders)
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")


# %% [markdown]
# ### Grid search to find best parameters to optimize fit with model

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Step 1: Log Transformation to Address Heteroskedasticity
monthly_orders_log = np.log(monthly_orders + 1)  # Adding 1 to avoid log(0)

# Step 2: Re-confirm Optimal Parameters Using Grid Search
import itertools

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
best_aic = float("inf")
best_pdq = None

for param in pdq:
    try:
        model = ARIMA(monthly_orders_log, order=param)
        model_fit = model.fit()
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_pdq = param
    except:
        continue

print(f"Optimal ARIMA parameters: {best_pdq} with AIC: {best_aic}")

# %%
# Step 3: Fit the Model with Optimal Parameters
optimal_order = best_pdq
model = ARIMA(monthly_orders_log, order=optimal_order)
model_fit = model.fit()

# Print Summary
print(model_fit.summary())

# Step 4: Check Residuals
residuals = model_fit.resid

# Plot Residuals
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(residuals)
plt.title("Residuals")
plt.subplot(212)
plt.hist(residuals, bins=20)
plt.title("Residual Histogram")
plt.tight_layout()
plt.show()

# Check Autocorrelation of Residuals
plot_acf(residuals)
plot_pacf(residuals)
plt.show()

# Ljung-Box Test for Residual Autocorrelation
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)

# Step 5: Forecast Future Values
num_steps = 10  # Adjust the number of steps to forecast
forecast = model_fit.forecast(steps=num_steps)

# Generate the correct time index for the forecast
last_date = monthly_orders.index[-1]
forecast_index = pd.date_range(start=last_date, periods=num_steps + 1, freq='M')[1:]  # Match the frequency of your data
forecast = pd.Series(forecast.values, index=forecast_index)

# Reverse log transformation for visualization (if applicable)
forecast_original = np.exp(forecast) - 1  # Reverse log transform

# Plot Forecast
plt.figure(figsize=(10, 6))
plt.plot(monthly_orders, label="Original Data")
plt.plot(forecast_original, label="Forecast", color="red")
plt.legend()
plt.title("Improved ARIMA Forecast")
plt.xlabel("Time")
plt.ylabel("Values")
plt.show()

# %%
#Forecasting the next 5 months for predicted distinct orders

# Forecasting the next 5 months
forecast_steps = 5
forecast = model_fit.forecast(steps=forecast_steps)

# Create a forecast index to align with forecasted values
forecast_index = pd.date_range(start=monthly_orders.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Convert forecast to a pandas Series with proper index
forecast_series = pd.Series(data=forecast, index=forecast_index)



# %%
# Plot historical data
plt.figure(figsize=(10, 6))
plt.plot(monthly_orders, label='Historical Data', color='blue', marker='o')

# Plot forecasted data
plt.plot(forecast_index, forecast, label='Forecast', color='red', marker='x')

# Add title, labels, and legend
plt.title('ARIMA Forecast for Distinct Orders')
plt.xlabel('Date')
plt.ylabel('Number of Orders')
plt.legend()
plt.grid()

# Show the plot
plt.show()



# %%
model_fit.plot_diagnostics(figsize=(12, 8))
plt.show()

# %%
# Assume `monthly_orders` is a pandas Series with time as the index
# Split into train and test sets
train_data = monthly_orders[:'2010-12']
test_data = monthly_orders['2011-01':]

# Train ARIMA model
model = ARIMA(train_data, order=(1, 1, 1))  # Example ARIMA parameters
model_fit = model.fit()

# Forecast for the test period
forecast = model_fit.forecast(steps=len(test_data))

# Extract actual values
y_actual = test_data.values  # Actual values from the test dataset
y_forecast = forecast.values  # Forecasted values from the ARIMA model

# Calculate MAPE
def calculate_mape(y_actual, y_forecast):
    import numpy as np
    y_actual, y_forecast = np.array(y_actual), np.array(y_forecast)
    return np.mean(np.abs((y_actual - y_forecast) / y_actual)) * 100

mape = calculate_mape(y_actual, y_forecast)
print(f"MAPE: {mape:.2f}%")

# %% [markdown]
# Extremely high MAPE score meeans the ARIMA model is performing poorly in forecasting our test data

# %% [markdown]
# ## SARIMA Model Implementation

# %%
df.info()

# %%

# Aggregate data: Count distinct orders per month
monthly_data = df.groupby('order_month').agg({
    'Customer Order Code': 'nunique'  # Count distinct orders
}).rename(columns={'Customer Order Code': 'Distinct Orders'})

# Convert index to datetime for SARIMA compatibility
monthly_data.index = monthly_data.index.to_timestamp()
print(monthly_data.head())

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

#Not enough observations


# Check stationarity and difference the data if necessary
result = adfuller(monthly_data['Distinct Orders'])
print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
if result[1] > 0.05:
    monthly_data['Stationary Orders'] = monthly_data['Distinct Orders'].diff().dropna()

# Fit SARIMA model
model = SARIMAX(monthly_data['Distinct Orders'], 
                order=(1, 1, 1),  # ARIMA parameters (p, d, q)
                seasonal_order=(1, 1, 1, 12),  # SARIMA parameters (P, D, Q, s)
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit()

# Print summary
print(model_fit.summary())

# %%
# Forecast next 12 months
forecast_steps = 12
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(
    start=monthly_data.index[-1], periods=forecast_steps + 1, freq='M'
)[1:]

# Forecast values
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Print forecast
print(forecast_mean)

# Plot forecast with confidence intervals
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Distinct Orders'], label='Historical Data')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, 
                 forecast_conf_int.iloc[:, 0], 
                 forecast_conf_int.iloc[:, 1], 
                 color='pink', alpha=0.3)
plt.legend()
plt.title("SARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Distinct Orders")
plt.show()

# %%

# Actual values (from the test set)
y_actual = monthly_data['Distinct Orders'][-12:]  # Replace -12 with the appropriate range of test data

# Forecasted values (from SARIMA model)
y_forecast = forecast.predicted_mean  # Replace with your forecasted values for the same test period

# Ensure both are aligned and converted to numpy arrays
y_actual = np.array(y_actual)
y_forecast = np.array(y_forecast)

# Compute MAPE
def calculate_mape(y_actual, y_forecast):
    # Avoid division by zero
    non_zero_mask = y_actual != 0
    return np.mean(np.abs((y_actual[non_zero_mask] - y_forecast[non_zero_mask]) / y_actual[non_zero_mask])) * 100

mape = calculate_mape(y_actual, y_forecast)
print(f"MAPE: {mape:.2f}%")

# %% [markdown]
# This is a horrible score.... SARIMA is not compatible with this dataset

# %%
df.head()

# %% [markdown]
# # Multi-Classification Model

# %% [markdown]
# Going to use Random Forest and XG Boost

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define categorical and numerical columns
categorical_features = ['Customer Country Code', 'Route', 'season']
numerical_features = ['items', 'value']
target = 'Product Code'

# Split dataset into features (X) and target (y)
X = df[categorical_features + numerical_features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', MinMaxScaler(), numerical_features)
    ]
)

# Random Forest pipeline
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy:.2f}")
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# %%
#Hyper Parameter Tuning for Random Forest

# Define parameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],       # Number of trees
    'classifier__max_depth': [None, 10, 20, 30],     # Maximum depth of each tree
    'classifier__min_samples_split': [2, 5, 10],     # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 4]        # Minimum samples required at a leaf node
}

# Initialize GridSearchCV with the rf_model pipeline
grid_search = GridSearchCV(estimator=rf_model,      # Use the existing pipeline
                           param_grid=param_grid, 
                           scoring='accuracy',      # Metric for evaluation
                           cv=5,                    # 5-fold cross-validation
                           verbose=1)               # Show progress

# Perform the grid search
grid_search.fit(X_train, y_train)

# Output the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)



# %%
#rf_model = Pipeline(steps=[('preprocessor', preprocessor), ('rf', RandomForestClassifier(max_depth= None, min_samples_leaf= 1, min_samples_split=2, n_estimators= 50))])

# %%
# Random Forest with best parameters

rf_model = Pipeline(steps=[('preprocessor', preprocessor), ('rf', RandomForestClassifier(max_depth= 10, min_samples_leaf= 4, min_samples_split=10, n_estimators= 50))])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print("Random Forest Classification Report: ")
print(classification_report(y_test, y_pred_rf))

# %% [markdown]
# ### Quantity Demanded XGB Regression Model

# %%
df.head()

# %%
# Define categorical and numerical columns
categorical_features = ['Customer Country Code', 'Product Code', 'Route', 'season']
numerical_features = ['value', 'Demand_lead_time']

# Target variable
target = 'items'

# Split dataset into features (X) and target (y)
X = df[categorical_features + numerical_features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', MinMaxScaler(), numerical_features)
    ]
)


# %%
#Converting all Nulls in item to 0s so the model can work...

df['items'] = df['items'].fillna(0)

# %%
import numpy as np
from sklearn.metrics import mean_squared_error

# Check and clean the target variable
y = y.replace([np.inf, -np.inf], np.nan).dropna()
X = X.loc[y.index]  # Ensure alignment

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', MinMaxScaler(), numerical_features)
    ]
)

# XGBoost model
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', 
                               n_estimators=100, 
                               max_depth=6, 
                               learning_rate=0.1, 
                               random_state=42))
])

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# %%
# Define parameter grid
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1, 0.2]
}

# GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", -grid_search.best_score_)

# %%
#Run XGB Regressor with best params

# Ensure no missing values
X_train = X_train.fillna(0)
y_train = y_train.fillna(0)

# Convert categorical columns to numeric using preprocessing pipeline
X_train = preprocessor.fit_transform(X_train)

# Ensure y_train is numeric
if y_train.dtypes == 'object':
    from sklearn.preprocessing import LabelEncoder
    y_train = LabelEncoder().fit_transform(y_train)

# Run the XGBRegressor with best parameters
best_xgb_model = XGBRegressor(
    n_estimators=best_params['regressor__n_estimators'],
    max_depth=best_params['regressor__max_depth'],
    learning_rate=best_params['regressor__learning_rate'],
    objective='reg:squarederror',
    random_state=42
)

# Fit the model
best_xgb_model.fit(X_train, y_train)

# Predict on the test set (make sure to preprocess X_test as well)
X_test = preprocessor.transform(X_test)
y_pred = best_xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error with Best Parameters: {mse:.2f}")

# %% [markdown]
# THIS MODEL POPPED OFF WITH THE ACCURACY

# %%
import matplotlib.pyplot as plt
import numpy as np

# Sort the actual and predicted values for better visualization
sorted_indices = np.argsort(y_test)
sorted_y_test = y_test.iloc[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(sorted_y_test.values, label="Actual Values", marker="o", color="blue")
plt.plot(sorted_y_pred, label="Predicted Values", marker="x", color="red")

# Add title, labels, and legend
plt.title("XGB Regressor: Actual vs Predicted Values")
plt.xlabel("Sample Index")
plt.ylabel("Quantity Demanded")
plt.legend()
plt.grid()

# Show the plot
plt.show()

# %%
# Scatter plot for Predicted vs Actual
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color="purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2)  # 45-degree line

# Add title and labels
plt.title("XGB Regressor: Predicted vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid()

# Show the plot
plt.show()

# %% [markdown]
# # Demand Lead Time for Orders

# %%
#Demand lead time in quarters

#I chose the quarter names arbitrarily since it wasn't clarified in the rubric
lead_time_quantiles = df['Demand_lead_time'].quantile([0.25, 0.5, 0.75])
print("Lead Time Quantiles:\n", lead_time_quantiles)

def categorize_lead_time(days):
    if days <= lead_time_quantiles[0.25]:
        return 'Short'
    elif days <= lead_time_quantiles[0.75]:
        return 'Medium'
    else:
        return 'Long'

df['Lead_Time_Category'] = df['Demand_lead_time'].apply(categorize_lead_time)

# Optional: Analyze lead time categories
print(df['Lead_Time_Category'].value_counts())

# %%
aggregated_df.head()


# %% [markdown]
# ### Monte Carlo

# %%
# Monte Carlo Sim for total Demand

n_simulations = 1000
simulated_demands = []
for _ in range(n_simulations):
    sampled_lead_times = aggregated_df['Demand_lead_time'].sample(len(aggregated_df), replace=True)
    sampled_quantities = aggregated_df['items'].sample(len(aggregated_df), replace=True)
    total_demand = (sampled_lead_times * sampled_quantities).sum()
    simulated_demands.append(total_demand)

# Analyze and visualize simulation results
simulated_demands = np.array(simulated_demands)
print("Simulated Total Demand (Mean):", simulated_demands.mean())
print("Simulated Total Demand (Std Dev):", simulated_demands.std())


# %%
# Place after Monte Carlo simulation is complete
plt.hist(simulated_demands, bins=50, alpha=0.75, color='blue')
plt.title('Monte Carlo Simulation: Total Demand Distribution')
plt.xlabel('Total Demand')
plt.ylabel('Frequency')
plt.show()

# Add confidence interval calculation
conf_interval = np.percentile(simulated_demands, [2.5, 97.5])
print("95% Confidence Interval:", conf_interval)


# %%
# Comparision of simulated advance demand to actual advance demand in test data
actual_advance_demand = aggregated_df['Demand_lead_time'].sum()  # Example: Adjust as per actual dataset
simulated_mean_demand = simulated_demands.mean()

print(f"Actual Advance Demand: {actual_advance_demand}")
print(f"Simulated Mean Advance Demand: {simulated_mean_demand}")

# Calculate percentage difference
percentage_difference = ((simulated_mean_demand - actual_advance_demand) / actual_advance_demand) * 100
print(f"Percentage Difference: {percentage_difference:.2f}%")



# %%
from scipy.stats import lognorm

# Fit lognormal distribution to demand lead time
shape, loc, scale = lognorm.fit(df['Demand_lead_time'])
print(f"Lognormal Parameters: Shape={shape}, Location={loc}, Scale={scale}")

# Plot distribution
sns.histplot(df['Demand_lead_time'], kde=True, stat='density')
x = np.linspace(df['Demand_lead_time'].min(), df['Demand_lead_time'].max(), 100)
plt.plot(x, lognorm.pdf(x, shape, loc, scale), label='Lognormal Fit', color='red')
plt.legend()
plt.show()


# %%
#As we can see from above this was completely pointless and transforming it doesn't do anything

# %% [markdown]
# # Final Edits

# %% [markdown]
# # HACKATHON STRATEGY

# %% [markdown]
# 1. Cleaning and EDA
# 
# 2. Group total transactions into months to gather the distict orders (uses aggregated df) - Monthly Info
# 
# 3. Specifically Prediction # of distinct orders per month using a time-series forecasting model (ARIMA / SARIMA)
# 
# 3. Develop Multi-Class Model to identify which products the customer will order based on historical patterns and SEASONALITY (utilitze OHE) 
#     - Using RF for classification
# 
# 4. Develop a predicting model for Quantity Demanded (items) for each product using XGB Regressor or quantiles (or other methods)
# 
# 5. Obtain Demand Lead Time using Monte Carlo Simulation, consolidating it into total demand
# 
# 6. Consolidate the models into demand forecasting, (Combining all the models into a final predictive forecasting model) - Used to quantify performance
# 
# 7. Scenario Analysis to evaluate how reducing demand time can impact the demand predictions 

# %% [markdown]
# 


