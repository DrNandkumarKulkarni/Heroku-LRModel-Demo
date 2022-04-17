import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('Ecommerce.csv')
df.shape
df.head(10)
df.columns
df.nunique(axis=0)
df.describe()
df[df.duplicated()].shape
df=df.dropna(axis=0)
df.shape
df_cleaned=df.copy().drop(['Customer ID'],axis=1)

df_cleaned.boxplot('Avg Session length')
Q1=df_cleaned['Avg Session length'].quantile(0.25)
Q3=df_cleaned['Avg Session length'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
median = df_cleaned.loc[df_cleaned['Avg Session length'].between(Lower_Whisker,Upper_Whisker), 'Avg Session length'].median()
median
df_cleaned.loc[df_cleaned['Avg Session length'] > Upper_Whisker, 'Avg Session length'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned.loc[df_cleaned['Avg Session length'] <Lower_Whisker, 'Avg Session length'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned

Q1=df_cleaned['Time on App'].quantile(0.25)
Q3=df_cleaned['Time on App'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
median = df_cleaned.loc[df_cleaned['Time on App'].between(Lower_Whisker,Upper_Whisker), 'Time on App'].median()
median
df_cleaned.loc[df_cleaned['Time on App'] > Upper_Whisker, 'Time on App'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned.loc[df_cleaned['Time on App'] <Lower_Whisker, 'Time on App'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned

Q1=df_cleaned['Time on Website'].quantile(0.25)
Q3=df_cleaned['Time on Website'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
median = df_cleaned.loc[df_cleaned['Time on Website'].between(Lower_Whisker,Upper_Whisker), 'Time on Website'].median()
median
df_cleaned.loc[df_cleaned['Time on Website'] > Upper_Whisker, 'Time on Website'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned.loc[df_cleaned['Time on Website'] <Lower_Whisker, 'Time on Website'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned

Q1=df_cleaned['Length of MemberShip'].quantile(0.25)
Q3=df_cleaned['Length of MemberShip'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
median = df_cleaned.loc[df_cleaned['Length of MemberShip'].between(Lower_Whisker,Upper_Whisker), 'Length of MemberShip'].median()
print(median)
df_cleaned.loc[df_cleaned['Length of MemberShip'] > Upper_Whisker, 'Length of MemberShip'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned.loc[df_cleaned['Length of MemberShip'] <Lower_Whisker, 'Length of MemberShip'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned

Q1=df_cleaned['Yealy amount spent'].quantile(0.25)
Q3=df_cleaned['Yealy amount spent'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)
median = df_cleaned.loc[df_cleaned['Yealy amount spent'].between(Lower_Whisker,Upper_Whisker), 'Yealy amount spent'].median()
print(median)
df_cleaned.loc[df_cleaned['Yealy amount spent'] > Upper_Whisker, 'Yealy amount spent'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned.loc[df_cleaned['Yealy amount spent'] <Lower_Whisker, 'Yealy amount spent'] = np.nan
df_cleaned.fillna(median,inplace=True)
df_cleaned

df_cleaned = df_cleaned.dropna(axis=0)
df_cleaned.shape

corr=df_cleaned.corr()

df_cleaned=df_cleaned.rename({'Avg Session length':'ASL', 'Time on App':'ToA', 'Time on Website':'ToW',
       'Length of MemberShip':'LoM', 'Yealy amount spent':'YAS'},axis=1)
df_cleaned.head()

#X = df_cleaned[['ASL','ToA','ToW','LoM']]
X=df_cleaned.iloc[:,:4]
X.head()

#y = df_cleaned[['YAS']]
y=df_cleaned.iloc[:,-1]
y.head()

LRmodel=LinearRegression()
LRmodel.fit(X,y)
pickle.dump(LRmodel,open('model.pkl','wb'))
