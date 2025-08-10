import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

df=pd.read_csv(r"E:\python\supervised learning\Polynomial\car.csv")


print(df.info())
# sns.boxplot(data=df)
# plt.show()

numeric_df = df.select_dtypes(include=['float64','int64'])

# Correlation with price
corr = numeric_df.corr()['price'].sort_values(ascending=False)

print(corr)
sns.heatmap(numeric_df.corr()[['price']], annot=True, cmap='coolwarm')
plt.show()
x=df[["symboling","wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg"]]
y=df["price"]

feature=df[["fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem"]]
ohe=OneHotEncoder(sparse_output=False,drop="first")
encod_array=ohe.fit_transform(feature)
getcolumns=ohe.get_feature_names_out(feature.columns)
encode_dataframe=pd.DataFrame(encod_array,columns=getcolumns)

le=LabelEncoder()
fea=df[["CarName"]]
lebal_aaray=le.fit_transform(fea)
leabadata=pd.DataFrame(lebal_aaray,columns=fea.columns)

x_final=pd.concat([x,encode_dataframe,leabadata],axis=1)
sc = StandardScaler()
X_scaled = sc.fit_transform(x_final)
pe=PolynomialFeatures(degree=1)
x_pe=pe.fit_transform(X_scaled)

x_train,x_test,y_train,y_test=train_test_split(x_pe,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test))

def user_input(symboling, wheelbase, carlength, carwidth, carheight, curbweight,
               enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm,
               citympg, highwaympg, fueltype, aspiration, doornumber, carbody, drivewheel,
               enginelocation, enginetype, cylindernumber, fuelsystem, CarName):
    
    # Numeric DataFrame
    numeric_input = pd.DataFrame([[symboling, wheelbase, carlength, carwidth, carheight,
                                   curbweight, enginesize, boreratio, stroke, compressionratio,
                                   horsepower, peakrpm, citympg, highwaympg]],
                                 columns=x.columns)
    
    # Categorical DataFrame
    cat_input = pd.DataFrame([[fueltype, aspiration, doornumber, carbody, drivewheel,
                               enginelocation, enginetype, cylindernumber, fuelsystem]],
                             columns=feature.columns)
    
    # Encode categorical
    cat_encoded = ohe.transform(cat_input)
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=getcolumns)
    
    # Encode CarName
    carname_encoded = le.transform([CarName])
    carname_df = pd.DataFrame(carname_encoded, columns=["CarName"])
    
    # Final combined input
    final_input = pd.concat([numeric_input, cat_encoded_df, carname_df], axis=1)
    
    # Scale + Polynomial Transform
    final_scaled = sc.transform(final_input)
    final_poly = pe.transform(final_scaled)
    
    # Prediction
    predicted_price = lr.predict(final_poly)[0]
    print("Predicted Car Price: $", round(predicted_price, 2))


# -----------------------------
# Taking input from user
# -----------------------------
symboling = int(input("Symboling: "))
wheelbase = float(input("Wheelbase: "))
carlength = float(input("Car length: "))
carwidth = float(input("Car width: "))
carheight = float(input("Car height: "))
curbweight = float(input("Curb weight: "))
enginesize = float(input("Engine size: "))
boreratio = float(input("Bore ratio: "))
stroke = float(input("Stroke: "))
compressionratio = float(input("Compression ratio: "))
horsepower = float(input("Horsepower: "))
peakrpm = float(input("Peak RPM: "))
citympg = float(input("City MPG: "))
highwaympg = float(input("Highway MPG: "))
fueltype = input(f"Fuel type {list(df['fueltype'].unique())}: ")
aspiration = input(f"Aspiration {list(df['aspiration'].unique())}: ")
doornumber = input(f"Door number {list(df['doornumber'].unique())}: ")
carbody = input(f"Car body {list(df['carbody'].unique())}: ")
drivewheel = input(f"Drive wheel {list(df['drivewheel'].unique())}: ")
enginelocation = input(f"Engine location {list(df['enginelocation'].unique())}: ")
enginetype = input(f"Engine type {list(df['enginetype'].unique())}: ")
cylindernumber = input(f"Cylinder number {list(df['cylindernumber'].unique())}: ")
fuelsystem = input(f"Fuel system {list(df['fuelsystem'].unique())}: ")
CarName = input(f"Car Name example {list(df['CarName'].unique())[:5]}: ")

# Call function
user_input(symboling, wheelbase, carlength, carwidth, carheight, curbweight,
           enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm,
           citympg, highwaympg, fueltype, aspiration, doornumber, carbody, drivewheel,
           enginelocation, enginetype, cylindernumber, fuelsystem, CarName)
