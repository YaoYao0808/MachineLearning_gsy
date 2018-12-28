import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
score = reg.score(X, y)
coef = reg.coef_
intercept_ = reg.intercept_
y_predict = reg.predict(np.array([[3, 5]]))

print(score)
print(y_predict)