import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
model = LogisticRegression()

df = pd.read_csv('titanic.csv')
plt.scatter(df['Pclass'], df['Fare'])
plt.title('Relationship between passenger fare and class', c='blue', fontweight='bold', fontsize=15)
plt.xlabel('Passenger Class', c='red', fontsize=13, fontweight='bold')
plt.ylabel('Ship Fare', c='red', fontsize=13, fontweight='bold')
plt.show()
