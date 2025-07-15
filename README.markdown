# Flight Delay Prediction Project

## Project Overview
This project aims to predict flight delays using machine learning techniques. By analyzing factors such as weather, carrier information, and flight history, the model identifies flights likely to be delayed, aiding in improved resource management and passenger satisfaction.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Data Loading and Exploration](#data-loading-and-exploration)
- [Model Development](#model-development)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this project, ensure you have Python installed. Clone the repository and install the required dependencies using the following commands:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Dependencies
The project requires the following Python packages, which can be installed via pip:

```bash
pip install ydata-profiling category_encoders==2.3.0 pandas numpy missingno matplotlib seaborn tensorflow scikit-learn
```

The Jupyter notebook includes the following imports:

- **Data Manipulation and Visualization**: `pandas`, `numpy`, `missingno`, `matplotlib`, `seaborn`
- **Machine Learning**: `sklearn` (for `train_test_split`, `GridSearchCV`, `cross_val_score`, `StandardScaler`, `RobustScaler`, `DecisionTreeClassifier`, `GaussianNB`, `LogisticRegression`)
- **Deep Learning**: `tensorflow.keras` (for `Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `Adam`, `BinaryCrossentropy`)
- **Data Encoding**: `category_encoders`
- **Profiling**: `ydata_profiling`

## Data Loading and Exploration
The dataset is loaded from Google Drive (mounted in the notebook) and includes features such as:
- Departure and arrival delays (`depdelay`, `arrdelay`)
- Flight details (`scheduleddepartdatetime`, `origin`, `dest`, `uniquecarrier`)
- Market and hub information (`marketshareorigin`, `marketsharedest`, `hhiorigin`, `hhidest`, etc.)
- Temporal features (`year`, `month`, `dayofmonth`, `dayofweek`, `scheduledhour`)
- Airport and airline hub classifications

Initial exploration involves generating a profile report using `ydata_profiling` to understand data distributions, missing values, and correlations.

## Model Development
The project employs multiple machine learning models, including:

1. **Decision Tree Classifier**: For interpretable decision-making.
2. **Gaussian Naive Bayes**: For probabilistic classification.
3. **Logistic Regression**: For linear decision boundaries.
4. **Neural Network (TensorFlow/Keras)**: A sequential model with dense layers, dropout, and batch normalization, optimized with the Adam optimizer and binary cross-entropy loss.

Data preprocessing includes:
- Encoding categorical variables using `category_encoders`.
- Scaling numerical features with `StandardScaler` or `RobustScaler`.
- Splitting data into training and testing sets using `train_test_split`.

The neural network model was evaluated, yielding:
- **Training**: Accuracy ~34.8%, Precision ~99.6%, Recall ~99.7%, Loss ~0.0274
- **Testing**: Accuracy ~34.7%, Precision ~99.6%, Recall ~99.7%, Loss ~0.0276

A plot of training and validation accuracy is generated to assess model performance over epochs.

## Results
The neural network model shows high precision and recall but relatively low accuracy, indicating potential class imbalance or overfitting. Further tuning (e.g., via `GridSearchCV`) and cross-validation (`cross_val_score`, `cross_validate`) are used to optimize model performance.

## Usage
To run the notebook:
1. Mount your Google Drive in the notebook to access the dataset.
2. Execute the cells sequentially to install dependencies, load data, preprocess, train models, and evaluate results.
3. Use the generated profile report and accuracy plots to analyze the data and model performance.

Example command to run the notebook:
```bash
jupyter notebook Air_Flight_Delay.ipynb
```

## Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes. Ensure your code follows the project's structure and includes appropriate documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.