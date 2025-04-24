# Fraud Detection Model

## Description
This project is a machine learning-based fraud detection system that aims to identify fraudulent transactions in a financial dataset. It uses various classification algorithms and data preprocessing techniques to analyze transaction data and predict fraud. The project is built using Python, and it includes data preprocessing, model training, and evaluation. The goal is to create an accurate model that can predict fraudulent activities in financial transactions.

## Task Objectives
- **Preprocess the Data**: Handle missing values, scale features, and prepare the dataset for modeling.
- **Model Training**: Train multiple machine learning models to detect fraudulent transactions, including Decision Trees, Random Forest, and Support Vector Machines (SVM).
- **Model Evaluation**: Evaluate the model using metrics such as accuracy, precision, recall, and F1-score.
- **Data Visualization**: Visualize the performance of the model using metrics and plots.
- **Improve Accuracy**: Experiment with various techniques, such as hyperparameter tuning and cross-validation, to enhance the model's accuracy.

## Project Structure
- `intern.py`: Main script where data preprocessing, model training, and evaluation are performed.
- `fraudTrain.csv`: Training dataset containing labeled transaction data.
- `fraudTest.csv`: Test dataset for evaluating the model.
- `requirements.txt`: List of Python dependencies required to run the project.
- `README.md`: Project documentation.

## Requirements
- Python 3.x
- Required Python libraries can be installed by running:

      pip install -r requirements.txt
 ## Steps to Run the Project
  - 1. Clone the repository:
       - git clone https://github.com/Akshayareddy12022006/fraud-detection-model.git
cd fraud-detection-model
- 2. Install the required dependencies

         pip install -r requirements.txt

- 3. Run the main script:

         python intern.py

- 4. Review the output:
     - After running the script, the model will output the accuracy, classification report, and any visualizations for fraud detection. It will also save any important results or model metrics.
- 5. Model Output:
     - The model will output the accuracy of the prediction, a detailed classification report, and graphs to visualize its performance. Here is an example of the output:
    
     -          Model Accuracy: 95.6%
                Classification Report:
                             precision    recall  f1-score   support

                   Fraud       0.97      0.94      0.95       500
               Non-Fraud       0.94      0.97      0.96       500
                accuracy                           0.96      1000
               macro avg       0.96      0.96      0.96      1000
            weighted avg       0.96      0.96      0.96      1000
##How It Works
- Data Preprocessing:

   - The intern.py script first loads the training and test datasets (fraudTrain.csv and fraudTest.csv).

   - It then handles missing values and scales the features using StandardScaler to ensure that the data is ready for modeling.

- Model Training:

  - The script trains multiple classification models, such as Decision Trees, Random Forest, and SVM, using the preprocessed data.

- Model Evaluation:

  - After training the models, the script evaluates them using standard metrics like accuracy, precision, recall, and F1-score.

  - It also visualizes the performance metrics with graphs to help you analyze the results.

- Fine-Tuning:

  - Hyperparameters are tuned to improve the model's accuracy.

   - Cross-validation is performed to ensure the model generalizes well to unseen data.
