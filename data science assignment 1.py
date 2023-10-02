import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')
data.columns
data.drop(['Position'],axis=1,inplace=True)

##########################################3

x=data.iloc[:,0:1]
y=data.iloc[:,1:2]

##############################################

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)

regressor.coef_
regressor.intercept_

##################################################

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('level versus salary')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()
###############################
from sklearn.preprocessing import PolynomialFeatures
x_poly=PolynomialFeatures(degree=1)
ploy_reg=x_poly.fit_transform(x)

from sklearn.preprocessing import PolynomialFeatures
x_poly=PolynomialFeatures(degree=3)
ploy_reg=x_poly.fit_transform(x)


poly_regressor=LinearRegression()

poly_regressor.fit(ploy_reg,y)

plt.scatter(x,y,color='red')
plt.plot(x,poly_regressor.predict(ploy_reg))
plt.title('level versus salary')
plt.xlabel('level')
plt.ylebel('Salary')
plt.show()
poly_regressor.coef_
poly_regressor.intercept_










	 