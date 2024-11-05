# Binary Classification Streamlit App 🍄

This Streamlit application classifies mushrooms as edible or poisonous using a selection of binary classification models. It is built as part of the CPE312 Intro to Data Science course in Computer Engineering at SWU, providing an interactive and educational experience in model training and evaluation.

## Features
- **Classifier Selection: Choose from Support Vector Machine (SVM), Logistic Regression, and Random Forest for binary classification.
- **Hyperparameter Customization: Adjust parameters for each classifier, allowing fine-tuning for optimal performance.
- **Performance Evaluation: Visualize metrics including Accuracy, Precision, Recall, Confusion Matrix, ROC Curve, and Precision-Recall Curve.
- **Dataset Display: View the mushroom dataset directly within the app.

## Getting Started

### Prerequisites
Ensure you have all the required libraries by installing dependencies:

```bash
pip install -r requirements.txt
```

### Running the App
To launch the app:

```bash
streamlit run app.py
```

### Project Structure

- app.py: Main Streamlit app file.
- data/: Directory containing the mushrooms.csv dataset.
- requirements.txt: Lists required dependencies.

### How to Use the App
1. Use the sidebar to select a model (SVM, Logistic Regression, or Random Forest).
2. Adjust model hyperparameters to customize training.
3. Click "Classify" to train the model, and view metrics in real-time for analysis.
4. Optional: Check the "Show raw data" box to explore the dataset.

### Dataset and Model Evaluation
The dataset used is the UCI Mushroom dataset, which contains descriptions of 23 mushroom species across 22 features. The app splits this data into training and testing sets, with options to tune each model’s hyperparameters for optimal classification results.

### Citations
1. Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T.
2. Build a Machine Learning Web App with Streamlit and Python. Coursera Project Network.

### Acknowledgments
This app was developed as a hands-on learning project for understanding binary classification in data science.

