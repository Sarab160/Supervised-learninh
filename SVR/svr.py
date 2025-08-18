import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

df=pd.read_csv(r"E:\python\supervised learning\SVR\rainfall.csv")
df["average_rain_fall_mm_per_year"] = pd.to_numeric(df["average_rain_fall_mm_per_year"], errors="coerce")


df["average_rain_fall_mm_per_year"]=df["average_rain_fall_mm_per_year"].fillna(df["average_rain_fall_mm_per_year"].mean())

print(df.info())

le=LabelEncoder()
df["Area"]=le.fit_transform(df["Area"])
x=df[["Area","Year"]]
y=df["average_rain_fall_mm_per_year"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

svr=SVR(kernel="linear")
svr.fit(x_train,y_train)

print(svr.score(x_test,y_test))

