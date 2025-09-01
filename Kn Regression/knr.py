import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,KFold,cross_val_score
from sklearn.preprocessing  import StandardScaler,PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
df=pd.read_csv(r"e:\python\supervised learning\Kn Regression\uSA_housin.csv")
# print(df.head())

# sns.boxplot(data=df)
# plt.show()

x=df[["Avg. Area Income","Avg. Area House Age","Avg. Area Number of Rooms","Avg. Area Number of Bedrooms","Area Population"]]
y=df["Price"]
pe=PolynomialFeatures(degree=2)
x_pe=pe.fit_transform(x)
ss=StandardScaler()
x_ss=ss.fit_transform(x_pe)

x_train,x_test,y_train,y_test=train_test_split(x_ss,y,test_size=0.2,random_state=42)
knr=KNeighborsRegressor(n_neighbors=10,p=2)
knr.fit(x_train,y_train)
print(knr.score(x_test,y_test))

param_grid1 = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],        # Number of neighbors to use
    'weights': ['uniform', 'distance'],                # How weights are assigned to neighbors
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Search algorithm
    'leaf_size': [10, 20, 30, 40, 50],                 # Leaf size for BallTree/KDTree
    'p': [1, 2]                                         # Power parameter for Minkowski distance
}

# gd=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=param_grid1)
# gd.fit(x_train,y_train)

# print(gd.best_params_)
# print(gd.best_score_)

# kf=KFold(n_splits=5)
# print(cross_val_score(KNeighborsRegressor(),x_train,y_train,cv=kf))

y_pred = knr.predict(x_test)
print("Test R²:", knr.score(x_test, y_test))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print(knr.score(x_train,y_train))



y_train_pred = knr.predict(x_train)
y_test_pred = knr.predict(x_test)

# Create a figure with two subplots
plt.figure(figsize=(12,5))

# Train set plot
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Train Set: R² = {knr.score(x_train, y_train):.4f}")

# Test set plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Test Set: R² = {knr.score(x_test, y_test):.4f}")

plt.tight_layout()
plt.show()