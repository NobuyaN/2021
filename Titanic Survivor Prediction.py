import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score

df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'
x = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(x, y)

# Add prediction down below
predict = list(x[0])
print(model.predict([predict]))

# 0 - Did not Survived
# 1 - Did Survived

# percentage of correct value
print(model.score(x, y))
# 80% of the correct values

y_pred = model.predict(x)
print('Precision Score: ', precision_score(y, y_pred))
print('Accuracy Score: ', accuracy_score(y, y_pred))
print('Recall Score: ', recall_score(y, y_pred))
print('F1 Score: ', f1_score(y, y_pred))
