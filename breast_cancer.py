
# coding: utf-8

# In[1]:

from sklearn import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix
cancers=datasets.load_breast_cancer()

feature_name=cancers['feature_names']
data=pd.DataFrame(cancers['data'])
data.columns = [feature_name]
target1=pd.DataFrame(cancers['target'])

target_name=pd.DataFrame(cancers['target_names'])


# In[2]:

data.head(4)
# feature_name
target1.head()


# In[3]:

#查看数据中有无NULL值
data[data.isnull().values==True]#无空值
# data.isnull().any()
# pd.DataFrame(data.min(axis=0))


# In[4]:

#查看标签情况
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
M,N=target1[0].value_counts()
plt.bar(0,M,width=0.5,color='c',label='M')
plt.bar(1,N,width=0.5,color='m',label='N')
plt.xlabel("label kinds")
plt.ylabel("Counts")
plt.xticks((0,1),(u'M',u'N'))
plt.legend()
plt.show()


# In[5]:

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


# In[6]:

import seaborn as sns
target1.columns=['diagnosis']
target1_=target1['diagnosis'].replace((0,1),("M","N"))
data_n1=pd.concat([target1_,data_n.iloc[:,11:19]],axis=1)#可以查看所有的小提琴图
data_n2=pd.melt(data_n1,id_vars="diagnosis",var_name="Feature",value_name="value")
plt.figure(figsize=(10,10))
sns.violinplot(x="Feature",y="value",hue="diagnosis",data=data_n2,split=True, inner="quart")
# sns.swarmplot(x="Feature", y="value", data=data_n2, color="m", alpha=.5)
plt.xticks(rotation=60)


# In[7]:

import seaborn as sns
target1.columns=['diagnosis']
target1_=target1['diagnosis'].replace((0,1),("M","N"))
data_n1=pd.concat([target1_,data_n.iloc[:,20:29]],axis=1)#可以查看所有的小提琴图
data_n2=pd.melt(data_n1,id_vars="diagnosis",var_name="Feature",value_name="value")
plt.figure(figsize=(10,10))
sns.violinplot(x="Feature",y="value",hue="diagnosis",data=data_n2,split=True, inner="quart")
# sns.swarmplot(x="Feature", y="value", data=data_n2, color="m", alpha=.5)
plt.xticks(rotation=60)


# In[8]:

sns.distplot(data_n['mean concavity'])#查看数据分布情况


# In[9]:

# 比较某两个特征之间的相关性
print(feature_name)

# sns.jointplot(data_n.loc[:,'mean concavity'],data_n.loc[:,'mean concave points'],kind="regg",color='m')#从小提琴上观察发现比较相近，姑且看一下相关系数。
sns.jointplot(x='mean concavity',y='mean concave points',data=data_n,kind="regg",color='m')

# data_n


# In[10]:

import seaborn as sns
target1.columns=['diagnosis']
target1_=target1['diagnosis'].replace((0,1),("M","N"))
data_n1=pd.concat([target1_,data_n.iloc[:,0:15]],axis=1)
data_n2=pd.melt(data_n1,id_vars="diagnosis",var_name="Feature",value_name="value")
plt.figure(figsize=(10,10))
# sns.violinplot(x="Feature",y="value",hue="diagnosis",data=data_n2,split=True, inner="quart")
sns.swarmplot(x="Feature", y="value",hue="diagnosis", data=data_n2)
plt.xticks(rotation=60)


# In[11]:

import seaborn as sns
target1.columns=['diagnosis']
target1_=target1['diagnosis'].replace((0,1),("M","N"))
data_n1=pd.concat([target1_,data_n.iloc[:,16:29]],axis=1)
data_n2=pd.melt(data_n1,id_vars="diagnosis",var_name="Feature",value_name="value")
plt.figure(figsize=(10,10))
# sns.violinplot(x="Feature",y="value",hue="diagnosis",data=data_n2,split=True, inner="quart")
sns.swarmplot(x="Feature", y="value",hue="diagnosis", data=data_n2)
plt.xticks(rotation=60)


# In[14]:

#比较所有特征之间的相关性
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data_n.corr(),annot=True,linewidths=.5,fmt='.2f',ax=ax)


# In[13]:

# sns.pairplot(data_n1.iloc[:,0:29],hue="diagnosis",size=3,diag_kind="kde")


# In[15]:

#降为处理
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
data_pca=pca.fit_transform(data)
plt.scatter(data_pca[:,0],data_pca[:,1],c=target1['diagnosis'],edgecolor='r',cmap="summer",alpha=0.4)


# In[16]:

pd.DataFrame(data_pca).head()


# In[17]:

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

