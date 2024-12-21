import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/DATASETS/Hitters.csv')
data.isnull()
data.describe()
data.dropna(inplace=True)
data = pd.get_dummies(data, drop_first=True)
data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = data.drop('Salary', axis=1)
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
linear_model = LinearRegression()
ridge_model = Ridge(alpha=0.5748)
lasso_model = Lasso(alpha=0.5748)
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)