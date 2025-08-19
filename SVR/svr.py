import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVR


df=pd.read_csv(r"E:\python\supervised learning\SVR\rainfall.csv")
print(df.columns)
# print(df[" area"].nunique())

df["average_rain_fall_mm_per_year"] = pd.to_numeric(df["average_rain_fall_mm_per_year"], errors="coerce")


df["average_rain_fall_mm_per_year"]=df["average_rain_fall_mm_per_year"].fillna(df["average_rain_fall_mm_per_year"].mean())

print(df.info())

le=LabelEncoder()
df[" area"]=le.fit_transform(df[" area"])
x=df[[" area","Year"]]
y=df["average_rain_fall_mm_per_year"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

svr=SVR(kernel="poly")
svr.fit(x_train,y_train)

print(svr.score(x_test,y_test))

param_grid = {
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1, 1],
    "epsilon": [0.01, 0.1, 0.5, 1],
    "degree": [2, 3, 4]  # used only if kernel='poly'
}

gd=RandomizedSearchCV(estimator=SVR(),param_distributions=param_grid,cv=5,n_iter=10)
gd.fit(x_train,y_train)

print(gd.best_params_)
print(gd.best_score_)




