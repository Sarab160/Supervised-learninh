import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv(r"E:\python\supervised learning\KN Classifier\Iris.csv")

# sns.pairplot(data=df,hue="Species")
# plt.show()
######for outlier
# sns.boxplot(data=df)
# plt.show()
x=df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=df["Species"]
le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

knc=KNeighborsClassifier(n_neighbors=10)

knc.fit(x_train,y_train)

print("Accuracy : ",knc.score(x_test,y_test)*100)
print(knc.score(x_train,y_train))

clf=confusion_matrix(y_test,knc.predict(x_test))
sns.heatmap(data=clf,annot=True)
plt.show()
print("Prescion score: ",precision_score(y_test,knc.predict(x_test),average="macro"))
print("Recall score: ",recall_score(y_test,knc.predict(x_test),average="macro"))
print("F1 score: ",f1_score(y_test,knc.predict(x_test),average="macro"))

def input_function(spl,spw,pl,pw):
    x_input=pd.DataFrame([[spl,spw,pl,pw]],columns=x.columns)
    
    prediction=knc.predict(x_input)
    pred=le.inverse_transform(prediction)
    print(pred[0])

SepalLengthCm = float(input("Enter Sepal Length (cm): "))
SepalWidthCm = float(input("Enter Sepal Width (cm): "))
PetalLengthCm = float(input("Enter Petal Length (cm): "))
PetalWidthCm = float(input("Enter Petal Width (cm): "))

input_function(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)