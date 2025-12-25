Airline Customer Booking Prediction ML Model

📌 Project Overview:

This project analyzes airline customer booking behavior and builds a machine learning model to predict whether a customer will complete a booking.
The focus of this project is not just model accuracy, but building a correct end-to-end ML pipeline, with special attention to data preprocessing, train–test splitting, and data leakage prevention.


Dataset size: 50,000+ customer records


A.) Objectives:

->Predict customer booking intent 
->Understand which factors most strongly influence booking decisions
->Learn and apply best practices in:
  Feature encoding,Train–test separation, Data leakage prevention, Model interpretability

B.) Project Structure & Thought Process

1. Data Exploration
-Verified dataset integrity (no missing values, no duplicates)
-Inspected data types to ensure numerical and categorical fields were correctly represented

2. Feature Preprocessing
Different preprocessing strategies were used based on feature type:
a.)Nominal Features - [sales_channel, trip_type]
Converted using manual one-hot encoding to retain interpretability
b.)Ordinal Features - [flight_day]
Encoded with meaningful numeric order (Mon = 1 … Sun = 7)
c.) High Cardinality Nominal Features - [route, booking_origin]
One-hot encoding was avoided due to very large unique values
Frequency encoding was chosen instead

3. Data Leakage Discovery & Fix
During preprocessing, a data leakage issue was intentionally explored and corrected.
-Initial mistake:
Frequency encoding was applied before splitting the data, allowing the model to see future information from the test set.
-Correct approach:
First shuffle and split the dataset into training (80%) and testing (20%)
Compute frequency encodings only on training data
Apply those frequencies to the test set
Assign unseen categories a value of 0
This correction ensures the model does not gain access to future information.
A detailed explanation of this mistake and fix is documented in:
data_leakage_explained.py

4. ML Model Training
Algorithm used: Random Forest Classifier
Why? 
-Handles mixed data types well
-Resistant to overfitting
-Provides feature importance (interpretability)

The model was trained using:
x_train: feature matrix
y_train: booking outcome labels

5. Model Evaluation
Evaluated on unseen test data
Achieved approximately 85% accuracy
Verified prediction distribution to ensure the model was not biased toward predicting only one class

6. Model Interpretation
Extracted feature importance scores from the trained Random Forest
Identified key drivers of booking behavior such as:  [Purchase lead time, Route demand]

Visualized feature importance for easier interpretation


C.) Key Insights

Customers booking further in advance are significantly more likely to complete bookings

Certain routes and origins show consistently higher conversion behavior

Interpretability was prioritized to extract actionable insights, not just predictions