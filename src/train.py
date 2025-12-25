from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(x_train, y_train):
    print("Training Random Forest...")
    # random_state=42: Locks the "randomness"
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)
    return clf

def evaluate_model(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    
    print()
    print("Accuracy:", accuracy_score(y_test, y_pred)*100, "%")
    print()
    
    print("Predicted customer bookings (1s): ", sum(y_pred))
    print("Predicted customer non-bookings (0s): ", len(y_pred)-sum(y_pred), "\n")

def get_feature_importance(clf, x_train):
    importance=clf.feature_importances_
    
    df_imp=pd.DataFrame({"Feature":x_train.columns, 
                         "Importance %":importance*100 })

    df_imp=df_imp.sort_values(by="Importance %", ascending=False)
    print(df_imp)
    return df_imp