# Used Laptop Price Estimator 💻
A Machine Learning project that predicts the estimated resale price of used laptops based on their hardware specifications, brand, operating system, storage configuration, and physical condition.
The project uses historical eBay laptop listings to train and evaluate regression models that estimate laptop prices in USD.

## Project Overview
Buying or selling a used laptop can be challenging because prices vary depending on hardware specifications, brand, and device condition.
This project explores how Machine Learning can be used to estimate the resale value of used laptops by analyzing their specifications and historical listing prices.

## Dataset
The dataset was obtained from eBay used laptop listings.
It contains information about laptop specifications, including:
- Brand
- Processor
- Screen Size
- RAM Size
- SSD Capacity
- Hard Drive Capacity
- GPU
- Processor Speed
- Resolution
- Operating System
- Storage Type
- Condition
- Price (Target Variable)

### Data Preprocessing
The dataset underwent several cleaning operations:
- Removed laptops listed as new.
- Removed listings categorized as "For parts or not working".
- Removed records with invalid RAM values below 1 GB.
- Removed irrelevant Item Number identifiers.
- Stripped unnecessary whitespace from text columns.
- Investigated inconsistencies in storage capacity information.
- Removed exact duplicate records.
- Investigated suspicious RAM specifications and excluded two suspected incorrect records.
**Final dataset:** 255 records and 17 columns.

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis was performed to understand the dataset and identify relationships between laptop specifications and prices.
The analysis included:
- Price distribution using histograms and boxplots.
- Numerical feature analysis and outlier investigation.
- Correlation analysis using a heatmap.
- RAM size versus laptop price.
- Laptop condition versus price.
- Price comparisons across laptop brands.
- Investigation of storage capacity inconsistencies.

## Machine Learning Models
Four regression algorithms were implemented and evaluated:
1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Random Forest Regressor

### Model Preprocessing
A Scikit-learn Pipeline and ColumnTransformer were used to handle preprocessing and model training.
- **StandardScaler:** Scales numerical features.
- **OneHotEncoder:** Converts categorical features into numerical representations.
- **Pipeline:** Combines preprocessing and model training into a single workflow.

### Model Evaluation
The models were evaluated using:
| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error. |
| RMSE | Measures prediction error while penalizing larger errors. |
| R² | Measures how much variation in laptop prices is explained by the model. |


## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/Astro790/used-laptop-price-estimator.git
```
### 2. Navigate to the Project Directory
```bash
cd used-laptop-price-estimator
```
### 3. Install Dependencies
pip install -r requirements.txt

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Git and GitHub

## Project Goals
- Apply Machine Learning to a real-world pricing problem.
- Understand and perform data preprocessing.
- Explore relationships between laptop specifications and resale prices.
- Compare different regression algorithms.
- Apply feature engineering and hyperparameter tuning.
- Build a reusable model for price prediction.

## Limitations
- The dataset contains a relatively small number of records.
- Predictions are based on historical eBay listings and may not reflect current market prices.
- Laptop specifications and listing information may contain inaccuracies.
- Estimated prices are not guaranteed selling prices.