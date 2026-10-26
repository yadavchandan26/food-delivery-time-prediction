<div align="center">

# 🛵 Food Delivery Time Prediction

**An end-to-end machine learning pipeline that predicts how long a food delivery will take, in minutes.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Compared-189AB4)
![Best Model](https://img.shields.io/badge/Best%20Model-Linear%20Regression-success)
![Test R²](https://img.shields.io/badge/Test%20R%C2%B2-0.832-brightgreen)
![RMSE](https://img.shields.io/badge/RMSE-9.16%20min-informational)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Highlights](#-key-highlights)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Model Comparison](#-model-comparison)
- [Sample Prediction](#-sample-prediction)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🎯 Overview

Accurate delivery estimates matter for both customers and delivery platforms. This project treats the problem as a **supervised regression task**: given an order's **distance, traffic level, weather, time of day, vehicle type, preparation time, and courier experience**, predict the **delivery time (`Delivery_Time_min`)**.

The repo covers the full ML workflow: data inspection, EDA, preprocessing, model training and tuning, model comparison, and prediction export.

---

## ✨ Key Highlights

- 🧹 **Clean preprocessing**: median / most-frequent imputation, one-hot encoding, standard scaling
- 🤖 **Six models compared**: Decision Tree, Linear Regression, Random Forest, Lasso, Ridge, XGBoost
- 🏆 **Linear Regression wins**: test **R² = 0.832**, **RMSE = 9.16 min**
- 🧱 **Modular code**: preprocessing, training, and evaluation live in separate modules
- 📤 **One-command pipeline**: `python main.py` goes from raw CSV to `output.csv`
- 📓 **Full notebook**: EDA and every experiment in `notebook_collection/`

---

## 📁 Project Structure

```
food-delivery-time-prediction/
│
├── data/
│   └── Food_Delivery_Times.csv     # Raw dataset
│
├── notebook_collection/            # EDA and model experiments (Jupyter)
│
├── prediction/                     # Modular pipeline code
│   ├── data_processing.py          # Imputation, encoding, scaling, train/test split
│   ├── model_training.py           # Model training
│   └── evaluate_model.py           # Scoring and prediction export
│
├── main.py                         # Entry point: runs the full pipeline
├── output.csv                      # Generated predictions
└── README.md
```

---

## 📊 Dataset

**Source:** [Food Delivery Time Prediction](https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction) on Kaggle  
**Target:** `Delivery_Time_min`

| Column | Type | Description |
|--------|------|-------------|
| `Order_ID` | int | Unique order identifier (dropped before training) |
| `Distance_km` | float | Distance between restaurant and customer |
| `Weather` | categorical | Clear, Rainy, Foggy, Snowy, Windy |
| `Traffic_Level` | categorical | Low, Medium, High |
| `Time_of_Day` | categorical | Morning, Afternoon, Evening, Night |
| `Vehicle_Type` | categorical | Bike, Scooter, Car |
| `Preparation_Time_min` | int | Time the restaurant takes to prepare the food |
| `Courier_Experience_yrs` | float | Courier's experience in years |
| `Delivery_Time_min` | int | **Target**: total delivery time in minutes |

**Data quality:** `Weather`, `Traffic_Level`, `Time_of_Day`, and `Courier_Experience_yrs` each have some missing values (3%).

---

## ⚙️ Pipeline

```
Raw CSV ─► Imputation ─► One-Hot Encoding ─► Train/Test Split ─► Scaling ─► Model Training ─► Evaluation ─► output.csv
```

| Step | What happens |
|------|--------------|
| **1. Imputation** | Median for numeric columns, most-frequent for categorical columns |
| **2. Encoding** | One-hot encoding (`drop_first=True`) on Traffic, Weather, Vehicle, Time of Day |
| **3. Split** | 70% train / 30% test, `random_state=42` |
| **4. Scaling** | `StandardScaler` fit on train, applied to test |
| **5. Training** | Fit the final Linear Regression model |
| **6. Export** | Predict on the test set and save to `output.csv` |

---

## 🏆 Model Comparison

All models were trained on the same 70/30 split. Decision Tree, Random Forest, and the final choice were tuned with `GridSearchCV` where applicable.

| Model | Train R² | Test R² | RMSE (min) |
|-------|:--------:|:-------:|:----------:|
| **Linear Regression** ⭐ | 0.748 | **0.832** | **9.16** |
| Ridge (α=1) | 0.748 | 0.832 | 9.16 |
| Lasso (α=1) | 0.719 | 0.793 | 10.18 |
| Random Forest | 0.945 | 0.792 | 10.21 |
| XGBoost | 0.938 | 0.786 | 10.34 |
| Decision Tree | 0.737 | 0.709 | 12.07 |

**What the results show**

- **Linear models win.** Linear Regression and Ridge are effectively tied, which suggests the relationship between the features and delivery time is largely linear.
- **Tree ensembles overfit.** Random Forest and XGBoost reach ~0.94 train R² but only ~0.79 on test.
- **Simple beats complex** on this small (1,000-row) tabular dataset, and the model stays easy to interpret.

---

## 🔮 Sample Prediction

```python
new_order = {
    "Distance_km": 8.5,
    "Weather": "Windy",
    "Traffic_Level": "Low",
    "Time_of_Day": "Evening",
    "Vehicle_Type": "Scooter",
    "Preparation_Time_min": 10,
    "Courier_Experience_yrs": 3.0,
}
# Predicted Delivery Time: 50.72 minutes
```

---

## ▶️ Usage

Run the full pipeline from the project root:

```bash
python main.py
```

This preprocesses the data, trains the model, prints the model score, and saves predictions to `output.csv`.

To explore the analysis and all experiments:

```bash
jupyter notebook notebook_collection/
```

---


## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data handling | Pandas, NumPy |
| Modelling | scikit-learn, XGBoost |
| Visualisation | Matplotlib, Seaborn |
| Experiments | Jupyter Notebook, Kaggle |


---

## 👤 Author

**Yadav Chandan**  
🔗 GitHub: [@yadavchandan26](https://github.com/yadavchandan26)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
