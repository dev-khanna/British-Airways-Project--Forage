import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(df_imp):
    plt.figure(figsize=(12,15))
    sns.barplot(x="Importance %", y="Feature", data=df_imp)
    plt.title("Factors Driving Booking Decisions")
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()