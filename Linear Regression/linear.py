import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

df=pd.read_csv(r"E:\python\supervised learning\Linear Regression\file/exams.csv")
print(df.info())

# plt.scatter(df["math score"],df["reading score"])
# plt.show()

x=df[["reading score","writing score"]]
y=df["math score"]

feature=df[["gender","race/ethnicity","parental level of education","lunch","test preparation course"]]
ohe=OneHotEncoder(sparse_output=False,drop="first",handle_unknown="ignore")
encode_array=ohe.fit_transform(feature)
getcolumns=ohe.get_feature_names_out(feature.columns)
encode_dataset=pd.DataFrame(encode_array,columns=getcolumns)

x_final=pd.concat([x,encode_dataset],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x_final,y,test_size=0.2,random_state=42)
lr=LinearRegression()

lr.fit(x_train,y_train)

print("Model Accuracy ",lr.score(x_test,y_test))

param_grid_lr = {
    'fit_intercept': [True, False],
    'positive': [True, False]  # Ensures coefficients are positive (optional)
}

gd=GridSearchCV(estimator=LinearRegression(),param_grid=param_grid_lr,cv=5)
gd.fit(x_train,y_train)
print("Result for Grid Search cv")
print(gd.best_params_)
print(gd.best_score_)
print("Result of cross validation kfold")
kf=KFold(n_splits=5)
scores=cross_val_score(LinearRegression(),x_train,y_train,cv=kf)

print(scores)
print("Result of Random Search Cv")
rd=RandomizedSearchCV(estimator=LinearRegression(),param_distributions=param_grid_lr,n_iter=20)
rd.fit(x_train,y_train)
print(rd.best_score_)

print("Mean squred error: ")
print(mean_squared_error(y_test,lr.predict(x_test)))


def user_input(gender,race,parental,lunch,preparation,reading,writing):
    input_feature=pd.DataFrame([[gender,race,parental,lunch,preparation]],columns=feature.columns)
    encode_array=ohe.transform(input_feature)
    input_encode_dataset=pd.DataFrame(encode_array,columns=getcolumns)
    
    numeric_input=pd.DataFrame([[reading,writing]],columns=x.columns)
    
    x_input=pd.concat([numeric_input,input_encode_dataset],axis=1)
    
    prediction=lr.predict(x_input)[0]
    print("Math score : ",prediction)
    

gender=input("Entert the gender male/female: ")
race=input("Enter the group group A/group B/group C/group D/group E: ")
par=input("Enter the parental level of eduction high school/some high school/some college/bachelor's degree/associate's degree: ")
lunch=input("Enter the lunch standard standard/(free/reduced): ")
pre=input("Enter preparation level completed/none: ")
reading=int(input("Enter the reading score: "))
writing=int(input("Enter the writing score: "))
user_input(gender,race,par,lunch,pre,reading,writing)

