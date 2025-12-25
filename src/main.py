import pandas as pd
from preprocessing import initial_preprocessing, frequency_encode_column
from train import train_model, evaluate_model, get_feature_importance
from visualization import plot_feature_importance
import os

# LOAD DATA
dir=os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(dir, '..', 'data', 'customer_booking.csv')
df=pd.read_csv(file_path) 

# PHASE 1 & 2: INITIAL PREPROCESSING
df = initial_preprocessing(df)

# PHASE 3: SPLIT
# Shuffle
df_shuffle=df.sample(frac=1, random_state=18)
X=df_shuffle.drop(columns=["booking_complete"])
Y=df_shuffle["booking_complete"]

# Cut + Assigned
length=int(0.8*len(df_shuffle))
x_train=X.iloc[:length].copy()
y_train=Y.iloc[:length].copy()
x_test=X.iloc[length:].copy()
y_test=Y.iloc[length:].copy()

# FREQUENCY ENCODING (Your route/origin logic)
# a.) route
x_train, x_test = frequency_encode_column(x_train, x_test, "route")
# b.) booking_origin
x_train, x_test = frequency_encode_column(x_train, x_test, "booking_origin")

print("Data preparation complete.")

# PHASE 4: MODELLING
clf = train_model(x_train, y_train)

# EVALUATION
evaluate_model(clf, x_test, y_test)

# PHASE 5: INTERPRETATION
df_imp = get_feature_importance(clf, x_train)

# PHASE 6: VISUALIZATION
plot_feature_importance(df_imp)