import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv(r"E:\python\supervised learning\Descision Tree Regressor\crop_yield.csv")

print(df.info())


# print(df["Crop"].unique())
# print(df["State"].nunique())
le=LabelEncoder()
df["Crop"]=le.fit_transform(df["Crop"])
df["State"]=le.fit_transform(df["State"])

x=df[["Crop","Crop_Year","State","Area","Production","Annual_Rainfall","Fertilizer","Pesticide"]]
y=df["Yield"]

feature=df[["Season"]]
ohe=OneHotEncoder(sparse_output=False,drop="first")
encode_array=ohe.fit_transform(feature)
get_name=ohe.get_feature_names_out(feature.columns)
encodedata=pd.DataFrame(encode_array,columns=get_name)

X=pd.concat([x,encodedata],axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

dtr=DecisionTreeRegressor(splitter= 'random', min_samples_split=10, min_samples_leaf= 4, max_features= None,max_depth= 50, criterion= 'squared_error')
dtr.fit(x_train,y_train)

print("Train score",dtr.score(x_train,y_train))
print("Test score",dtr.score(x_test,y_test))
print("Mean squared error",mean_absolute_error(y_test,dtr.predict(x_test)))
print("Mean absolute error",mean_absolute_error(y_test,dtr.predict(x_test)))

param_grid = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'auto', 'sqrt', 'log2']
}

rd=RandomizedSearchCV(estimator=DecisionTreeRegressor(),param_distributions=param_grid,n_iter=5)
rd.fit(x_train,y_train)
print(rd.best_params_)
print(rd.best_score_)
# train_scores = []
# test_scores = []

# for i in range(1, 30):
#     dtr1 = DecisionTreeRegressor(max_depth=i, random_state=42)
#     dtr1.fit(x_train, y_train)

#     # Append scores
#     train_scores.append(dtr1.score(x_train, y_train))
#     test_scores.append(dtr1.score(x_test, y_test))

# # Plot gantel (dumbbell) style
# plt.figure(figsize=(10,6))

# for i in range(len(train_scores)):
#     plt.plot([train_scores[i], test_scores[i]], [i+1, i+1], 'o-', color="gray")  # line between train & test
#     plt.scatter(train_scores[i], i+1, color="green", label="Train" if i==0 else "")
#     plt.scatter(test_scores[i], i+1, color="blue", label="Test" if i==0 else "")

# plt.yticks(range(1,30), range(1,30))
# plt.xlabel("R² Score")
# plt.ylabel("Max Depth")
# plt.title("Decision Tree Regressor Performance (Train vs Test)")
# plt.legend()
# plt.grid(True)
# plt.show()
    
    