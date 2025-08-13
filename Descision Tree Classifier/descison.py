import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

df=pd.read_csv(r"E:\python\supervised learning\Descision Tree Classifier\ads.csv")
print(df.head())

print(df.info())

sns.pairplot(data=df,hue="Purchased")
plt.show()

x=df[["Age","EstimatedSalary"]]
y=df["Purchased"]

feature=df[["Gender"]]
ohe=OneHotEncoder(sparse_output=False,drop="first")
encodearray=ohe.fit_transform(feature)
getcolumn=ohe.get_feature_names_out(feature.columns)

encode_dataframe=pd.DataFrame(encodearray,columns=getcolumn)

X=pd.concat([x,encode_dataframe],axis=1)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

dtc=DecisionTreeClassifier(class_weight= 'balanced', criterion= 'entropy', max_depth= 20, max_features= None, min_samples_leaf= 1, min_samples_split= 20, splitter= 'random')
dtc.fit(x_train,y_train)

print(dtc.score(x_test,y_test))
print(dtc.score(x_train,y_train))

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],      # Splitting criteria
    'splitter': ['best', 'random'],                    # Split strategy
    'max_depth': [None, 5, 10, 15, 20, 25],            # Tree depth
    'min_samples_split': [2, 5, 10, 20],               # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4, 6],                  # Minimum samples at a leaf node
    'max_features': [None, 'sqrt', 'log2'],            # Features to consider for split
    'class_weight': [None, 'balanced'],                # Handle imbalance
}

# gd=GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=param_grid,cv=2)
# gd.fit(x_train,y_train)

# print(gd.best_params_)
# print(gd.best_score_)
plot_tree(dtc)
plt.show()