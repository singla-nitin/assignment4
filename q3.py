from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
california = fetch_california_housing()
X = california.data
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ridge = Ridge(alpha=0.5748)
ridge_cv_scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
ridge_cv_mse = -np.mean(ridge_cv_scores)

ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_test_mse = mean_squared_error(y_test, ridge_pred)

lasso = Lasso(alpha=0.5748)
lasso_cv_scores = cross_val_score(lasso, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
lasso_cv_mse = -np.mean(lasso_cv_scores)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_test_mse = mean_squared_error(y_test, lasso_pred)

print("Ridge Coefficients:")
print(ridge.coef_)

print("Lasso Coefficients:")
print(lasso.coef_)

print(f'RidgeCV Cross-Validated MSE: {ridge_cv_mse}')
print(f'Ridge Test MSE: {ridge_test_mse}')
print(f'LassoCV Cross-Validated MSE: {lasso_cv_mse}')
print(f'Lasso Test MSE: {lasso_test_mse}')

if ridge_test_mse < lasso_test_mse:
    print("Best Method: Ridge Regression")
    print(f'Best Test MSE: {ridge_test_mse}')
else:
    print("Best Method: Lasso Regression")
    print(f'Best Test MSE: {lasso_test_mse}')