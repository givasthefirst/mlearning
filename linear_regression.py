import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Assigning a pandas dataframe to bmi_life_data variable.
# Loading the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')


# Train the model with the data 
# Make and fit the linear regression model
bmi_life_model = LinearRegression()

x_values = bmi_life_data[['Life expectancy']]
y_values = bmi_life_data[['BMI']]

bmi_life_model.fit(y_values, x_values)

# Predicting using the model
# Predict life expectancy for a BMI value of 55.0987
laos_life_exp = bmi_life_model.predict(55.0987)

print(laos_life_exp)

plt.scatter(y_values, x_values)
plt.plot(y_values, bmi_life_model.predict(y_values))

plt.title("Life expectancy for a given BMI")
plt.xlabel("BMI")
plt.ylabel("Life expectancy")
plt.show()





