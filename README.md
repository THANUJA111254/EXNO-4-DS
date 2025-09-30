# EXNO:4-DS
## REG NO:212224040231

# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
 import pandas as pd
 import numpy as np
 import seaborn as sns
 from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import accuracy_score, confusion_matrix
 data=pd.read_csv("income.csv",na_values=[ " ?"])
 data
```

<img width="1656" height="820" alt="image" src="https://github.com/user-attachments/assets/e8a5c76b-9fd2-4260-8fdb-38851730abc1" />

```
 data.isnull().sum()
```

<img width="277" height="663" alt="image" src="https://github.com/user-attachments/assets/819ab8ce-0e4e-4a29-9a60-b32cbc85ca1a" />


```
 missing=data[data.isnull().any(axis=1)]
 missing
```
<img width="1675" height="784" alt="image" src="https://github.com/user-attachments/assets/9a782bb1-29d2-41c1-b4db-fdd6748393c8" />


```
data2=data.dropna(axis=0)
data2
```

<img width="1687" height="770" alt="image" src="https://github.com/user-attachments/assets/f7489c17-74b0-4965-a303-8fef446a8583" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="961" height="295" alt="image" src="https://github.com/user-attachments/assets/09fc045a-94cc-4d58-b3f1-5b52b19fc24c" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="581" height="566" alt="image" src="https://github.com/user-attachments/assets/2b499d7d-88a7-4bac-845a-7ea15bdfa5d6" />

```
data2
```

<img width="1722" height="795" alt="image" src="https://github.com/user-attachments/assets/143d1775-70e4-47e5-8b8c-e50bce4e7c27" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1702" height="694" alt="image" src="https://github.com/user-attachments/assets/caf63493-5d15-4e4a-b6c8-49a72d279cbf" />


```
 columns_list=list(new_data.columns)
 print(columns_list)
```

<img width="1713" height="55" alt="image" src="https://github.com/user-attachments/assets/843dc770-0c33-4a48-9c02-40e39ee2e0ae" />

```
 features=list(set(columns_list)-set(['SalStat']))
 print(features)
```

<img width="1668" height="46" alt="image" src="https://github.com/user-attachments/assets/90ad5a0a-6fc8-464f-b56d-740d365d8673" />

```
 y=new_data['SalStat'].values
 print(y)
```

<img width="206" height="39" alt="image" src="https://github.com/user-attachments/assets/5a80a706-69df-47e6-8dbb-90c7388999fc" />

```
 x=new_data[features].values
 print(x)
```

<img width="474" height="203" alt="image" src="https://github.com/user-attachments/assets/6acd3c20-8248-4cbb-996f-a91319ab99c6" />

```
 train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
 KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
 KNN_classifier.fit(train_x,train_y)
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)
```

<img width="204" height="81" alt="image" src="https://github.com/user-attachments/assets/d2e1ed30-2216-4d4c-80ef-1ad432987f90" />

```
 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)
```

<img width="261" height="41" alt="image" src="https://github.com/user-attachments/assets/e8f29c4c-0c13-4d06-b744-3f9768ba9a56" />

```
 print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="432" height="47" alt="image" src="https://github.com/user-attachments/assets/4d93cb32-ef2a-4ca2-be19-c77e3b5b74b0" />

```
data.shape
```

<img width="153" height="31" alt="image" src="https://github.com/user-attachments/assets/6fb46e93-d332-4369-ad7d-42197664aa1b" />

```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
 data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 selected_feature_indices=selector.get_support(indices=True)
 selected_features=x.columns[selected_feature_indices]
 print("Selected Features:")
 print(selected_features)
```


<img width="404" height="56" alt="image" src="https://github.com/user-attachments/assets/274b4ae9-3b4c-4cc9-bc30-03ee847b663d" />

```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```

<img width="592" height="289" alt="image" src="https://github.com/user-attachments/assets/e1fc2337-6092-4171-a82c-b2bba9306427" />

```
tips.time.unique()
```
<img width="601" height="78" alt="image" src="https://github.com/user-attachments/assets/feece9ec-4a1b-45ca-bc0d-bfbc7d19887c" />

```
 contingency_table=pd.crosstab(tips['sex'],tips['time'])
 print(contingency_table)
```
<img width="291" height="140" alt="image" src="https://github.com/user-attachments/assets/578a46ab-f2f5-48f7-ab45-834e268e3112" />

```
 chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statisti
 cs: {chi2}")
 print(f"P-Value: {p}")
```
<img width="507" height="99" alt="image" src="https://github.com/user-attachments/assets/4059d9a5-1aef-45b0-bf92-6a3338e1a932" />






















# RESULT:

Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
