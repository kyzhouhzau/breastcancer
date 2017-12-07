

```python
from sklearn import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix
cancers=datasets.load_breast_cancer()

feature_name=cancers['feature_names']
data=pd.DataFrame(cancers['data'])
data.columns = [feature_name]
target1=pd.DataFrame(cancers['target'])

target_name=pd.DataFrame(cancers['target_names'])
```


```python
data.head(4)
# feature_name
target1.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看数据中有无NULL值
data[data.isnull().values==True]#无空值
# data.isnull().any()
# pd.DataFrame(data.min(axis=0))
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 30 columns</p>
</div>




```python
#查看标签情况
%matplotlib inline
import matplotlib.pyplot as plt 
M,N=target1[0].value_counts()
plt.bar(0,M,width=0.5,color='c',label='M')
plt.bar(1,N,width=0.5,color='m',label='N')
plt.xlabel("label kinds")
plt.ylabel("Counts")
plt.xticks((0,1),(u'M',u'N'))
plt.legend()
plt.show()
```


![png](output_3_0.png)



```python
#分析数据与类别的相关性
import seaborn as sns
# data_n=(data-data.mean())/(data.std())
data_n=((data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0)))#归一化
target1.columns=['diagnosis']
target1_=target1['diagnosis'].replace((0,1),("M","N"))
data_n1=pd.concat([target1_,data_n.iloc[:,0:10]],axis=1)
data_n2=pd.melt(data_n1,id_vars="diagnosis",var_name="Feature",value_name="value")
plt.figure(figsize=(10,10))
sns.violinplot(x="Feature",y="value",hue="diagnosis",data=data_n2,split=True, inner="quart")
# sns.swarmplot(x="Feature", y="value", data=data_n2, color="m", alpha=.5)
plt.xticks(rotation=60)

```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), <a list of 10 Text xticklabel objects>)




![png](output_4_1.png)



```python
import seaborn as sns
target1.columns=['diagnosis']
target1_=target1['diagnosis'].replace((0,1),("M","N"))
data_n1=pd.concat([target1_,data_n.iloc[:,11:19]],axis=1)#可以查看所有的小提琴图
data_n2=pd.melt(data_n1,id_vars="diagnosis",var_name="Feature",value_name="value")
plt.figure(figsize=(10,10))
sns.violinplot(x="Feature",y="value",hue="diagnosis",data=data_n2,split=True, inner="quart")
# sns.swarmplot(x="Feature", y="value", data=data_n2, color="m", alpha=.5)
plt.xticks(rotation=60)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7]), <a list of 8 Text xticklabel objects>)




![png](output_5_1.png)



```python
import seaborn as sns
target1.columns=['diagnosis']
target1_=target1['diagnosis'].replace((0,1),("M","N"))
data_n1=pd.concat([target1_,data_n.iloc[:,20:29]],axis=1)#可以查看所有的小提琴图
data_n2=pd.melt(data_n1,id_vars="diagnosis",var_name="Feature",value_name="value")
plt.figure(figsize=(10,10))
sns.violinplot(x="Feature",y="value",hue="diagnosis",data=data_n2,split=True, inner="quart")
# sns.swarmplot(x="Feature", y="value", data=data_n2, color="m", alpha=.5)
plt.xticks(rotation=60)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8]), <a list of 9 Text xticklabel objects>)




![png](output_6_1.png)



```python
sns.distplot(data_n['mean concavity'])#查看数据分布情况
```




    <matplotlib.axes._subplots.AxesSubplot at 0x222852597b8>




![png](output_7_1.png)



```python
# 比较某两个特征之间的相关性
print(feature_name)

# sns.jointplot(data_n.loc[:,'mean concavity'],data_n.loc[:,'mean concave points'],kind="regg",color='m')#从小提琴上观察发现比较相近，姑且看一下相关系数。
sns.jointplot(x='mean concavity',y='mean concave points',data=data_n,kind="regg",color='m')

# data_n
```

    ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean smoothness' 'mean compactness' 'mean concavity'
     'mean concave points' 'mean symmetry' 'mean fractal dimension'
     'radius error' 'texture error' 'perimeter error' 'area error'
     'smoothness error' 'compactness error' 'concavity error'
     'concave points error' 'symmetry error' 'fractal dimension error'
     'worst radius' 'worst texture' 'worst perimeter' 'worst area'
     'worst smoothness' 'worst compactness' 'worst concavity'
     'worst concave points' 'worst symmetry' 'worst fractal dimension']
    




    <seaborn.axisgrid.JointGrid at 0x2229ad1a4e0>




![png](output_8_2.png)



```python
import seaborn as sns
target1.columns=['diagnosis']
target1_=target1['diagnosis'].replace((0,1),("M","N"))
data_n1=pd.concat([target1_,data_n.iloc[:,0:15]],axis=1)
data_n2=pd.melt(data_n1,id_vars="diagnosis",var_name="Feature",value_name="value")
plt.figure(figsize=(10,10))
# sns.violinplot(x="Feature",y="value",hue="diagnosis",data=data_n2,split=True, inner="quart")
sns.swarmplot(x="Feature", y="value",hue="diagnosis", data=data_n2)
plt.xticks(rotation=60)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]),
     <a list of 15 Text xticklabel objects>)




![png](output_9_1.png)



```python
import seaborn as sns
target1.columns=['diagnosis']
target1_=target1['diagnosis'].replace((0,1),("M","N"))
data_n1=pd.concat([target1_,data_n.iloc[:,16:29]],axis=1)
data_n2=pd.melt(data_n1,id_vars="diagnosis",var_name="Feature",value_name="value")
plt.figure(figsize=(10,10))
# sns.violinplot(x="Feature",y="value",hue="diagnosis",data=data_n2,split=True, inner="quart")
sns.swarmplot(x="Feature", y="value",hue="diagnosis", data=data_n2)
plt.xticks(rotation=60)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
     <a list of 13 Text xticklabel objects>)




![png](output_10_1.png)



```python
#比较所有特征之间的相关性
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data_n.corr(),annot=True,linewidths=.5,fmt='.2f',ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2229cfea4a8>




![png](output_11_1.png)



```python
# sns.pairplot(data_n1.iloc[:,0:29],hue="diagnosis",size=3,diag_kind="kde")
```


```python
#降为处理
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
data_pca=pca.fit_transform(data)
plt.scatter(data_pca[:,0],data_pca[:,1],c=target1['diagnosis'],edgecolor='r',cmap="summer",alpha=0.4)

```




    <matplotlib.collections.PathCollection at 0x2229d6419b0>




![png](output_13_1.png)



```python
pd.DataFrame(data_pca).head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1160.142574</td>
      <td>-293.917544</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1269.122443</td>
      <td>15.630182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>995.793889</td>
      <td>39.156743</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-407.180803</td>
      <td>-67.380320</td>
    </tr>
    <tr>
      <th>4</th>
      <td>930.341180</td>
      <td>189.340742</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
logis=LogisticRegression(C=1000.0,random_state=0)
y_pred_logc=logis.fit(data_pca[:400],cancers['target'][:400])
pred=y_pred_logc.predict_proba(pd.DataFrame(data_pca[401:]))
fpr,tpr,threadhold=roc_curve(target1["diagnosis"][401:],pd.DataFrame(pred[:,1]))
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
print(auc(fpr,tpr))
```

    0.982186234818
    


![png](output_15_1.png)

