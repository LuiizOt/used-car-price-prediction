# Used Car Price Prediction

## Project Overview

Rusty Bargain, a used car sales service, is building an app in which users can quickly find out the market value of their car. Using about 354,000 historical listings with technical specifications and prices, this project trains and compares four regression models, evaluating them on the three criteria that matter to the business: **prediction quality**, **training time** and **prediction speed**.

**Key outcome:** Random Forest achieved the lowest error (RMSE ≈ 2,053, R² ≈ 0.80) but took about 6 minutes to train, while **LightGBM reached almost the same quality (RMSE ≈ 2,108, R² ≈ 0.78) in under 1 second**, which makes it the best quality-versus-speed trade-off.

## Business Problem

A price-estimation feature in a consumer app must be accurate enough to be trusted and fast enough to be practical. A model that is slightly more accurate but hundreds of times slower to train makes retraining on new listings costly. The model choice therefore has to balance accuracy with computational cost.

## Objectives

- Predict the market price of a used car from its characteristics.
- Compare models on validation RMSE and R².
- Measure training time and select the model with the best balance between quality and speed.

## Dataset Description

`car_data.csv`, provided for educational purposes as part of the TripleTen Data Science program, contains 354,369 listings and 16 columns.

| Group | Columns |
|---|---|
| **Target** | `Price` |
| Vehicle characteristics | `VehicleType`, `RegistrationYear`, `RegistrationMonth`, `Gearbox`, `Power`, `Model`, `Mileage`, `FuelType`, `Brand`, `NotRepaired` |
| Listing metadata | `DateCrawled`, `DateCreated`, `NumberOfPictures`, `PostalCode`, `LastSeen` |

**Data quality: missing values**

| Column | Missing values |
|---|---|
| `NotRepaired` | 71,154 |
| `VehicleType` | 37,490 |
| `FuelType` | 32,895 |
| `Gearbox` | 19,833 |
| `Model` | 19,705 |

## Technologies and Libraries

- Python
- pandas
- scikit-learn (`LinearRegression`, `GradientBoostingRegressor`, `RandomForestRegressor`, `train_test_split`, `root_mean_squared_error`, `r2_score`)
- LightGBM (`LGBMRegressor`)
- Jupyter Notebook

## Methodology

1. **Preprocessing:** column names are converted to lowercase. The listing metadata (`datecrawled`, `datecreated`, `lastseen`, `numberofpictures`) and `registrationmonth` are dropped as not useful for predicting price.
2. **Encoding:** categorical features are one-hot encoded with `pd.get_dummies(drop_first=True)`. Missing categories are not imputed; they become all-zero dummy columns. The resulting LightGBM training matrix has 265,776 rows and 291 used features.
3. **Split:** 75% training and 25% validation (`random_state=42`).
4. **Models:**
   - Linear Regression as a baseline;
   - scikit-learn Gradient Boosting (10 trees, depth 3, learning rate 0.1);
   - Random Forest (200 trees, depth 10);
   - LightGBM (200 trees, depth 3, 7 leaves, learning rate 0.05).
5. **Evaluation:** RMSE and R² on the validation set; training time measured with `time.time()` around `fit()`.

## Results and Model Performance

All metrics are computed on the **validation set**.

| Model | Training time (s) | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.98 | 3,288.38 | 0.475 |
| Gradient Boosting (scikit-learn) | 8.86 | 3,070.81 | 0.542 |
| Random Forest | 354.07 | **2,053.29** | **0.795** |
| **LightGBM** | **0.71** | 2,107.65 | 0.784 |

RMSE is expressed in the same unit as `Price`.

## Key Findings

- **Tree ensembles clearly beat the linear baseline:** Random Forest and LightGBM reduce the RMSE by about 36–38% relative to Linear Regression and raise R² from 0.48 to about 0.78–0.80. This suggests strongly non-linear relationships between car characteristics and price.
- **The Gradient Boosting configuration was too small:** with only 10 trees, scikit-learn's Gradient Boosting improves only slightly over the baseline.
- **LightGBM offers the best trade-off:** its RMSE is only about 54 higher than Random Forest's (≈ 2.6%), while it trains roughly 500 times faster (0.71 s vs. 354 s).

## Conclusions

**Achievement of objectives:** prediction quality and training time were compared for four models. LightGBM is the most suitable choice for the app given the stated criteria; Random Forest is marginally more accurate but far more expensive to train.

**Limitations**

- **Prediction speed was not measured**, although it is one of the business criteria.
- Results come from a single training/validation split; there is no separate test set and no cross-validation.
- The data was not checked for implausible values (for example, `Power` = 0 appears in the first record), and no outlier treatment was applied to `price`, `registrationyear` or `power`.
- Missing categorical values are not imputed. Combined with `drop_first=True`, a missing value produces the same all-zero encoding as the dropped reference category, so the two cases cannot be distinguished.
- `postalcode` is used as a numeric feature, although it is a categorical identifier.
- Hyperparameters were set manually rather than tuned.
- The notebook does not contain a written conclusion or a final model selection cell; the analysis above is based on the printed metrics.

**Potential improvements:** time `predict()` for each model, clean anomalous values, encode missing categories explicitly (for example, as an `unknown` category), treat `postalcode` as categorical or remove it, use LightGBM's native categorical handling instead of one-hot encoding, tune hyperparameters with cross-validation, and evaluate the final model on a held-out test set.

## Repository Structure

```text
used-car-price-prediction/
├── README.md
├── notebook.ipynb
└── datasets/
    └── car_data.csv
```

## How to Run

1. Create and activate a Python 3 environment (the notebook was last run with Python 3.13):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install the dependencies. `root_mean_squared_error` requires **scikit-learn 1.4 or later**:

   ```bash
   pip install pandas "scikit-learn>=1.4" lightgbm jupyter
   ```

3. Start Jupyter **from the project root**, so that the relative path `./datasets/car_data.csv` resolves correctly, and open `notebook.ipynb`:

   ```bash
   jupyter notebook notebook.ipynb
   ```

**Execution notes**

- `car_data.csv` is about 39 MB.
- One-hot encoding produces a wide feature matrix, and the Random Forest took about 6 minutes to train in the original run. Expect a similar duration and a substantial memory footprint.

## Author

**Luiz Otávio Carneiro Filho**
Geospatial Engineer | GIS & Data Science
[LinkedIn](https://www.linkedin.com/in/luizot/)
