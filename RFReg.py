import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px





df = pd.read_csv('CarPrice_Assignment.csv')

pd.set_option('display.max_columns' , 100)
print(df.head())

# #info about dataset
# print(df.shape)
# print(df.info())
# #data types in dataset
# for i in df.columns:
#   print(f"{i} : {df[i].dtype}")


#creating 2 lists for categorical and numerical data
cat_columns = []
num_columns = []
for i in df.columns:
  if df[i].dtype == object:
    cat_columns.append(i)
  else:
    num_columns.append(i)

print(df[cat_columns].nunique())


#Let’s also know the distribution of values in the categorical columns using value_counts() method.

# print(df['fueltype'].value_counts())
# print('------------------------------')
# print(df['aspiration'].value_counts())
# print('------------------------------')
# print(df['doornumber'].value_counts())
# print('------------------------------')
# print(df['carbody'].value_counts())
# print('------------------------------')
# print(df['drivewheel'].value_counts())
# print('------------------------------')
# print(df['enginelocation'].value_counts())
# print('------------------------------')
# print(df['enginetype'].value_counts())
# print('------------------------------')
# print(df['cylindernumber'].value_counts())
# print('------------------------------')
# print(df['fuelsystem'].value_counts())



#If you take a close look at the “CarName” column mostly the first 5 character seems to match.
#We reduced the unique value from 147 to 31

df['CarName'] = df['CarName'].str[:5]
print(df['CarName'].head(40))


#Handling the Categorical Variables
df1 = df.copy()

df1.replace({'fueltype':{'gas':0, 'diesel':1}}, inplace=True)
df1.replace({'aspiration':{'std':0, 'turbo':1}}, inplace=True)
df1.replace({'doornumber':{'four':0, 'two':1}}, inplace=True)
df1.replace({'enginelocation':{'front':0, 'rear':1}}, inplace=True)
df1.replace({'drivewheel':{'fwd':0, 'rwd':1, '4wd':2}}, inplace=True)
df1.replace({'carbody':{'sedan':0, 'hatchback':1, 'wagon':2, 'hardtop':3, 'convertible':4}}, inplace=True)
df1.replace({'cylindernumber':{'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}}, inplace=True)


pd.set_option('display.max_columns', 100)
print(df1.head())


df1_cat_columns = []
df1_num_columns = []
for i in df1.columns:
  if df1[i].dtype == object:
    df1_cat_columns.append(i)
  else:
    df1_num_columns.append(i)

print(df1[df1_cat_columns])

#Use LabelEncoder to convert the above columns into numbers

le = LabelEncoder()
df1[df1_cat_columns] = le.fit_transform(df1_cat_columns)
print(df1[df1_cat_columns])


#train_test_split
y = df1['price']
X = df1.drop(columns=['price', 'car_ID'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=786)


#model
#95% accuracy
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)


#metrics

rfr_mae = mean_absolute_error(y_test, rfr_prediction)
print(rfr_mae) # 1341.689531707317

rfr_r2 = r2_score(y_test, rfr_prediction)
print(rfr_r2 )# 0.9558504366285837

fig = px.scatter(x=y_test, y=rfr_prediction, labels={'x': 'Actual Price', 'y': 'Predicted Price'},  title='Random Forest Regressor Model Prediction')
fig.show()


