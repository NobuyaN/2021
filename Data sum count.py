import pandas as pd
df = pd.read_csv('titanic.csv')
arr = df[['Survived', 'Pclass', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values

male = df['Sex'] == 'male'
female = df['Sex'] == 'female'
child = df['Age'] < 18
total_fare = df['Fare']
survived = df['Survived'] == 1

print(f'Total Female on Board: {female.sum()}')
print(f'Total Male on Board: {male.sum()}')
print(f'Total Children on Board: {child.sum()}')
print(f'Total Passenger Survived: {survived.sum()}')
print(f'Total Fare of all passenger: {round(total_fare.sum())} dollars')
