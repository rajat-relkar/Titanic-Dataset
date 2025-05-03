# Titanic Survival Prediction

## Project Overview
This project uses the Titanic dataset to predict passenger survival using a K-Nearest Neighbors (KNN) classifier. The dataset is preprocessed, features are engineered, and the model is tuned for optimal performance. The final model achieves an accuracy of 80.22% on the test set.

## Dataset
The dataset (`titanic.csv`) contains information about Titanic passengers, including features like age, sex, passenger class, and fare. The target variable is `Survived`, indicating whether a passenger survived (1) or not (0).

## Requirements
To run the project, install the required Python libraries:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Project Structure
- `titanic.csv`: The dataset used for training and testing.

- `titanic_survival_prediction.py`: The main Python script containing the code for data preprocessing, feature engineering, model training, and evaluation.

- `README.md`: This file, providing an overview of the project.

## Code Description
The script performs the following steps:

1. **Imports**: Libraries for data manipulation (`pandas`, `numpy`), visualization (`seaborn`, `matplotlib`), and machine learning (`scikit-learn`).

2. **Data Loading**: Loads the Titanic dataset and checks for missing values.

3. **Data Preprocessing**:
   - Drops irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`, `Embarked`).

   - Fills missing `Age` values with the median age of the passenger's class.

   - Converts `Sex` to numerical values (`male`: 1, `female`: 0).

5. **Feature Engineering**:
   - Creates `FamilySize` by combining `SibSp` and `Parch`.

   - Adds `IsAlone` to indicate if a passenger is traveling alone.

   - Bins `Fare` into 4 categories (`FareBin`).

   - Bins `Age` into 5 categories (`AgeBin`).

7. **Machine Learning**:
   - Splits data into training (70%) and test (30%) sets.

   - Scales features using `MinMaxScaler`.

   - Tunes a KNN classifier using `GridSearchCV` to find the best hyperparameters (`n_neighbors`, `metric`, `weights`).

   - Evaluates the model using accuracy and a confusion matrix.

9. **Visualization**: Plots a heatmap of the confusion matrix to visualize model performance.

## How to Run

1. Ensure the required libraries are installed.

2. Place `titanic.csv` in the same directory as the script.

3. Run the script:
   ```bash
   python titanic_survival_prediction.py
   ```
4. The script outputs:
   - Data information and missing value counts.

   - Model accuracy (e.g., 80.22%).

   - Confusion matrix.

   - A plotted heatmap of the confusion matrix.

## Results

- **Accuracy**: 80.22%

- **Confusion Matrix**:

  ![](https://github.com/user-attachments/assets/213150e6-1d26-495e-8fa1-e939d931ad89)
  
  - True Negatives: 144 (correctly predicted not survived).

  - False Positives: 13 (incorrectly predicted survived).

  - False Negatives: 40 (incorrectly predicted not survived).

  - True Positives: 71 (correctly predicted survived).

## Future Improvements

- Experiment with other algorithms (e.g., Random Forest, Logistic Regression).

- Handle missing values in `Embarked` instead of dropping the column.

- Explore additional feature engineering, such as title extraction from `Name`.

- Address class imbalance in the target variable.
