import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv("Iris.csv")

print(df.head())

x=df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

y=df["Species"]

le=LabelEncoder()
y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
hnb=GaussianNB(var_smoothing= 1e-12)

hnb.fit(x_train,y_train)

print(hnb.score(x_test,y_test))
print(hnb.score(x_train,y_train))

param_grid = {
    "var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
}

gd=GridSearchCV(estimator=GaussianNB(),param_grid=param_grid)
gd.fit(x_train,y_train)

print(gd.best_params_)
print(gd.best_score_)