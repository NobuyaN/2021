import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
x = df[cancer_data.feature_names].values
y = df['target'].values

# 0 is malignant
# 1 is benign

model.fit(x, y)
print(model.predict([x[0]]))
print(f'The accuracy of this model is around {round(100 * (model.score(x, y)))}%')
