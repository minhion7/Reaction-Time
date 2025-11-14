# Reaction Time Prediction (ML Project)

Using Python, pandas, and scikit-learn, this project creates a basic machine learning model to **predict human reaction time** on cognitive tasks.

It pairs well with my Audio Emotion Recognizer project as a second example of applying ML to **cognitive science / human behavior data**.

---

## Files

- `reaction_time_project.ipynb` – main Jupyter/Colab notebook with all code
- `reaction_time.csv` – synthetic reaction-time dataset
- `screenshots/` – plots from the analysis (optional)

---

## Goal

Predict **reaction_time (ms)** from variables such as:

- `age`
- `trial_difficulty` (easy / medium / hard)
- `stimulus_type` (visual / audio)
- `congruency` (congruent / incongruent)

This simulates a classic experimental-psychology task where reaction time depends on task difficulty and stimulus factors.

---

## Methods

1. **Data loading & exploration**
   - Loaded `reaction_time.csv` into pandas
   - Inspected distributions and summary statistics
   - Plotted a histogram of reaction times

2. **Preprocessing**
   - One-hot encoded categorical variables with `pd.get_dummies`
   - Split data into train/test sets (80% / 20%)

3. **Models trained**
   - `LinearRegression` (baseline)
   - `RandomForestRegressor` (non-linear model with 200 trees)

4. **Evaluation metric**
   - Root Mean Squared Error (RMSE) on the test set

---

## Results (example)

On a typical run:

- With a lower RMSE than **Linear Regression**, the **Random Forest** model was better able to capture non-linear relationships between task variables and reaction time.

Plots include:

- Reaction time distribution histogram  
- Actual vs Predicted reaction times for the Random Forest model

(See the `screenshots/` folder.)

---

## Takeaways

- Reaction time can be reasonably predicted from a small set of experimental/task features.
- Non-linear models like Random Forests often perform better than simple linear models on behavioral data.
- This project demonstrates:
  - basic data handling in pandas  
  - feature encoding for ML  
  - training and evaluating multiple models in scikit-learn  
  - interpreting ML results in a **cognitive science** context

---

## Author

**Minh Vu**  
UC San Diego – Cognitive Science (Machine Learning focus)
