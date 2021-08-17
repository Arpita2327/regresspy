from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_iris
from regresspy.regression import Regression
from regresspy.loss import rmse

iris = load_iris()
X = iris.data[:, 0].reshape(-1, 1)
Y = iris.data[:, 1].reshape(-1, 1)

stochasticGradientDescent = SGDRegressor(max_iter= 100, learning_rate= 'constant', eta0= 0.001)
stochasticGradientDescent.fit(X, Y.reshape(-1))
stochasticGradientDescentPrediction = stochasticGradientDescent.predict(X)
stochasticGradient_rmse = rmse(stochasticGradientDescentPrediction, Y)
print('Stochastic Gradient Descent Regression RMSE value:', str(stochasticGradient_rmse))

regressionValue = Regression(epochs= 100, learning_rate= 0.0001)
regressionValue.fit(X, Y)
regressionPrediction = regressionValue.predict(X)
regression_rmse_value = regressionValue.score(regressionPrediction, Y)
print('RMSE value of class: ', str(regression_rmse_value))