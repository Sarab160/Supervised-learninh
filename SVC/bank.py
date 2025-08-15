import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score
from sklearn.svm import SVC
df=pd.read_csv(r"E:\python\supervised learning\SVC\bank.csv")

# print(df.head())
print(df.info())




# sns.pairplot(data=df)
# plt.show()
q1=df["balance"].quantile(0.25)
q3=df["balance"].quantile(0.75)

iqr=q3-q1

lower=q1-1.5*iqr
upper=q3+1.5*iqr
# outliers= df[(df["balance"] >= lower) & (df["balance"] <= upper)]
df = df[(df["balance"] >= lower) & (df["balance"] <= upper)].reset_index(drop=True)
# print("Number of outliers in balance:", len(df))

print(df.shape)

# sns.boxplot(data=df)
# plt.show()

x=df[["age","balance","day","duration","campaign","pdays","previous"]]


le=LabelEncoder()
df["deposit"]=le.fit_transform(df["deposit"])

Y=df["deposit"]

feature=df[["job","marital","education","default","housing","loan","contact","month","poutcome"]]
ohe=OneHotEncoder(sparse_output=False,drop="first")
encode_array=ohe.fit_transform(feature)
getcolumns=ohe.get_feature_names_out(feature.columns)
encode_dataframe=pd.DataFrame(encode_array,columns=getcolumns)

X=pd.concat([x,encode_dataframe],axis=1)
ss=StandardScaler()
X_f=ss.fit_transform(X)

x_train,x_test,y_train,y_test=train_test_split(X_f,Y,test_size=0.2,random_state=42)

svc=SVC(kernel="linear")
svc.fit(x_train,y_train)

print(svc.score(x_test,y_test))

param_grid = {
    'C': [0.1, 1, 10, 100],                # Regularization strength
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4, 5],                # Only for 'poly'
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],  # 'rbf', 'poly', 'sigmoid'
    'coef0': [0, 0.1, 0.5, 1],             # 'poly', 'sigmoid'
    'shrinking': [True, False],            # Whether to use shrinking heuristic
    'probability': [True],                 # Enable probability estimates (slower)
    'tol': [1e-3, 1e-4, 1e-5],              # Tolerance for stopping
    'max_iter': [-1, 1000, 5000]           # Limit on iterations (-1 = no limit)
}


# gd=GridSearchCV(estimator=SVC(),param_grid=param_grid)
# gd.fit(x_train,y_train)
# print(gd.best_params_)
# print(gd.best_score_)

