#!/usr/bin/env python
# coding: utf-8

#  
# #  Is it possible to predict popular perfumes?
# 

# # Settings:

# In[6]:


import bs4
from splinter import Browser
from bs4 import BeautifulSoup
import time
from webdriver_manager.chrome import ChromeDriverManager
import requests
import pandas as pd
import re
import selenium
from pandas import DataFrame
import json
from selenium import webdriver
import scipy as sc
import numpy as np
import time
import os
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement


# # Collect 6000 links of perfumes

# In[419]:


driver = webdriver.Chrome()

links_list=list()
for i in range(1,100):
    
    driver.get('https://www.fragrancex.com/shopping/best?instock=false&currentPage=' + str(i) + '&searchSortExpression=0')
    driver.implicitly_wait(10)
    more_items_elements = driver.find_elements(By.CLASS_NAME,'search-result-grid')

    for more_items_element in more_items_elements:
        link_elements = more_items_element.find_elements(By.CSS_SELECTOR, 'a.link-2')
        for link_element in link_elements:
            links_list.append(link_element.get_attribute('href'))

len(links_list)


# In[479]:


print(len(links_list))
driver.close()


# # CREATING ALL LIST'S:

# In[481]:


####### CREATING ALL LIST'S #######
brand_name_LIST=[]
fragrance_family_LIST=[]
fragrance_name_LIST=[]
fragrance_classification_LIST=[]
volume_LIST=[]
top_notes_LIST=[]
heart_notes_LIST=[]
base_notes_LIST=[]
gender_LIST=[]
ingredients_LIST=[]
product_form_LIST=[]
year_of_launch_LIST = []
strength_LIST = []
sustainable_LIST = []
country_of_origin_LIST = []

reviews_LIST = []
grade_LIST = []

grade_5_stars_LIST = []
grade_4_stars_LIST = []
grade_3_stars_LIST = []
grade_2_stars_LIST = []
grade_1_stars_LIST = []

recommend_to_friend_LIST=[]


# # Crawling:

# In[484]:


from selenium.common.exceptions import NoSuchElementException
sum_perfume_succeeded=0
driver2 = webdriver.Chrome()

for i in links_list:
    
    driver2.get(i)
    driver2.implicitly_wait(30)
    time.sleep(2)
        
    # Find all tr elements on the page
    table = driver2.find_element(By.TAG_NAME,'tbody')
    elements = table.find_elements(By.TAG_NAME,'tr')
    
    # Create empty lists to store the text from each td element
    name_elements = []
    text_elements = []

    # Iterate through the tr elements and extract the text from each td element
    for element in elements:
        tds = element.find_elements(By.TAG_NAME, 'td')
        name_elements.append(tds[0].text)
        text_elements.append(tds[1].text)
    
    #This code will check if the string "...." is in the name_elements list and if yes insert to ""...."_name_LIST else NAN
    try:
        index = name_elements.index("Brand")
        brand_name_LIST.append(text_elements[index])
    except ValueError:
        brand_name_LIST.append(None)
        
    try:
        index = name_elements.index("Fragrance Family")
        fragrance_family_LIST.append(text_elements[index])
    except ValueError:
        fragrance_family_LIST.append(None)
        
    
    try:
        index = name_elements.index("Fragrance Name")
        fragrance_name_LIST.append(text_elements[index])
    except ValueError:
        fragrance_name_LIST.append(None)

    try:
        index = name_elements.index("Fragrance Classification")
        fragrance_classification_LIST.append(text_elements[index])
    except ValueError:
        fragrance_classification_LIST.append(None)
       
    try:
        index = name_elements.index("Volume")
        volume_LIST.append(text_elements[index])
    except ValueError:
        volume_LIST.append(None)

    
    try:
        index = name_elements.index("Top Notes")
        top_notes_LIST.append(text_elements[index])
    except ValueError:
        top_notes_LIST.append(None)
    
    
    try:
        index = name_elements.index("Heart Notes")
        heart_notes_LIST.append(text_elements[index])
    except ValueError:
        heart_notes_LIST.append(None)

        
    try:
        index = name_elements.index("Base Notes")
        base_notes_LIST.append(text_elements[index])
    except ValueError:
        base_notes_LIST.append(None)
        
        
    try:
        index = name_elements.index("Gender")
        gender_LIST.append(text_elements[index])
    except ValueError:
        gender_LIST.append(None)
    

    try:
        index = name_elements.index("Product Form")
        product_form_LIST.append(text_elements[index])
    except ValueError:
        product_form_LIST.append(None)

        
    try:
        index = name_elements.index("Year Of Launch")
        year_of_launch_LIST.append(text_elements[index])
    except ValueError:
        year_of_launch_LIST.append(None)

        
    try:
        index = name_elements.index("Strength")
        strength_LIST.append(text_elements[index])
    except ValueError:
        strength_LIST.append(None)
 
    try:
        index = name_elements.index("Sustainable")
        sustainable_LIST.append(text_elements[index])
    except ValueError:
        sustainable_LIST.append(None)

          
    try:
        index = name_elements.index("Country of Origin")
        country_of_origin_LIST.append(text_elements[index])
    except ValueError:
        country_of_origin_LIST.append(None)
        
        
    try:
        index = name_elements.index("Ingredients")
        ingredients_LIST.append(text_elements[index])
    except ValueError:
        ingredients_LIST.append(None)
       
    
    button = driver2.find_element(By.CSS_SELECTOR, "a.btn-type-2.write-review")
    if (button.text=="Write a Review"):
            
        rating_element = driver2.find_element(By.CSS_SELECTOR, 'div[itemprop="ratingValue"][class="h2 serif header-large"]')
        grade_LIST.append(rating_element.text)
            
        review_count_element = driver2.find_element(By.CSS_SELECTOR, 'div[itemprop="reviewCount"][class="review-count"]')
        review_count = review_count_element.get_attribute('content')
        reviews_LIST.append(review_count)
           
    else:
        grade_LIST.append(None)
        reviews_LIST.append(None)

    sum_perfume_succeeded = sum_perfume_succeeded+1
    print(sum_perfume_succeeded)
    #if(sum_perfume_succeeded==100):
    #    driver2.quit()
driver2.quit()
print(sum_perfume_succeeded)


# In[ ]:


print(len(strength_LIST))
print(len(brand_name_LIST))
print(len(grade_LIST))
print(len(reviews_LIST))
print(len(recommend_to_friend_LIST))


# # Crawling for recommend_to_friend_LIST :

# In[842]:


driver2 = webdriver.Chrome()

k=1

for i in links_list:
    print(k)
    k=k+1
    if(k>=3242):
        
        driver2.get(i)
        driver2.implicitly_wait(30)
        time.sleep(1)
        try:
            button = driver2.find_element(By.CSS_SELECTOR, "a.btn-type-2.write-review")
    
            if (button.text=="Write a Review"):
        
                recommend_to_friend = driver2.find_elements(By.CSS_SELECTOR, 'div[class="h2 serif header-large"]')
                recommend_to_friend_LIST.append(recommend_to_friend[1].text)
           
            else:
                recommend_to_friend_LIST.append(None) 
        except:
            recommend_to_friend_LIST.append(None)
driver2.quit()


# # Create a DataFrame:

# In[848]:


df = pd.DataFrame({'Brand':brand_name_LIST,'fragrance_name':fragrance_name_LIST,'gender':gender_LIST,'reviews':reviews_LIST,'grade':grade_LIST,'recommend_to_friend':recommend_to_friend_LIST,'fragrance_family':fragrance_family_LIST,
                   'fragrance_classification':fragrance_classification_LIST,'volume':volume_LIST,'top_notes':top_notes_LIST,
                  'heart_notes':heart_notes_LIST,'base_notes':base_notes_LIST,
                  'product_form':product_form_LIST,'year_of_launch':year_of_launch_LIST,'strength':strength_LIST,
                  'sustainable':sustainable_LIST,'country_of_origin':country_of_origin_LIST,'ingredients':ingredients_LIST})
df


# # create copy of df

# In[850]:


data_fram_2 = df.copy()
data_fram_2


# # Create a CSV file & read to df_read

# In[ ]:


#data_fram_2.to_csv('perfiumes_data.csv',index=False)


# In[194]:


df_read=pd.read_csv('perfiumes_data.csv')


# In[195]:


df_read


# # normalization 

# In[196]:


df_final = df_read.copy()
  
#normalization 
column = 'grade'
df_final[column] = (df_final[column] - df_final[column].min()) / (df_final[column].max() - df_final[column].min())    
  
# view normalized data
#display(df_normalized)
df_final['grade']


# In[197]:


df_final['grade'].rank()


# # remove the rows that do not have a grade and reviews

# In[198]:


# remove the lines that do not have a grade and reviews
df_final=df_final.dropna(subset=['grade'])

#Rearranges the indexes again
df_final.reset_index(inplace=True,drop=True)


# In[199]:


df_final.info()


# # Deleting columns that do not have enough data:
# According to the info, we see that data is missing in:
# * ingredients               
# * country_of_origin         
# * sustainable               
# * strength 
# * product_form              

# In[200]:


df_final.drop(['ingredients'],axis=1,inplace=True)
df_final.drop(['country_of_origin'],axis=1,inplace=True)
df_final.drop(['sustainable'],axis=1,inplace=True)
df_final.drop(['strength'],axis=1,inplace=True)
df_final.drop(['product_form'],axis=1,inplace=True)
df_final


# # drop duplicates

# In[201]:


df_final.drop_duplicates(inplace=True)


# In[202]:


df_final.describe(include='all')   


# In[203]:


df_final["Brand"] = df_final["Brand"].astype("category")
df_final["fragrance_name"] = df_final["fragrance_name"].astype("string")

df_final["gender"] = df_final["gender"].astype("category")
df_final["reviews"] = df_final["reviews"].astype("int")

df_final["recommend_to_friend"] = df_final["recommend_to_friend"].astype("category")
df_final["fragrance_family"] = df_final["fragrance_family"].astype("category")
df_final["fragrance_classification"] = df_final["fragrance_classification"].astype("category")
df_final["volume"] = df_final["volume"].astype("category")
df_final["top_notes"] = df_final["top_notes"].astype("category")
#df_final["year_of_launch"] = df_final["year_of_launch"].astype("category")
df_final.info()


# In[ ]:





# # -------------------------------------- EDA - visualizations-------------------------------------------

# In[204]:


from matplotlib import pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # find and fix outliers 

# In[205]:


plt.hist(df_final.year_of_launch,bins=50)
plt.xlabel('year')
plt.ylabel('frequency')


# In[142]:


# Compute the 25th and 75th percentiles
q25 = df_final["year_of_launch"].quantile(0.25)
q75 = df_final["year_of_launch"].quantile(0.75)

# Calculate the IQR
iqr = q75 - q25

# Identify values that are more than 1.5 times the IQR below the 25th percentile or above the 75th percentile
outliers = df_final[(df_final["year_of_launch"] < q25 - 3*iqr)
                    | (df_final["year_of_launch"] > q75 + 3*iqr)].index

# drop outliers

for i in outliers:
    df_final = df_final.drop(i, axis=0)


# In[143]:


# Same histogram
plt.hist(df_final.year_of_launch,bins=50)
plt.xlabel('year')
plt.ylabel('frequency')


# # --------------------------------------------------------- (no more outliers)

# # Scatter_plot  (year of launch , grade)

# In[144]:


plt.scatter(df_final.year_of_launch,df_final.grade)


# # Grades histogram:

# In[145]:


fig=plt.figure(figsize=(12,4))

fig1=fig.add_subplot(1,2,1)
#fig2=fig.add_subplot(1,2,2)

fig1.hist(df_final.grade,bins=50)
fig1.set_title('grades histogram')

#fig2.hist(df_final.Brand,bins=30)
#fig2.set_title('Brand histogram')


# # Histogram - Grade by gender ( Men , Women , Men and Women)

# In[146]:


grouped_df = df_final.groupby('gender')
for gen,group in grouped_df:
    group['grade'].hist(bins=30,edgecolor='black',figsize=(18,4))
    plt.title(gen)
    plt.show()


# # From this histogram we understand that unisex perfumes will receive a higher grade.

# # Barplot: mean grade for all fragrance family

# In[206]:


grouped_df = df_final.groupby('fragrance_family')

means = grouped_df["grade"].mean()

sns.barplot(x=means.index,y=means.values)
plt.xticks(rotation=90)

plt.show()


# # From this barplot we understand that 
#  * floral, fruity and musk
#  * woody,oriental and fresh notes 
#  
# perfumes will receive a higher grade.
# 

# # Scatter_plot :
# # A grade for each company and the number of its perfumes on the site

# In[148]:


mean_grades = df_final.groupby("Brand")["grade"].mean()
# Count the number of occurrences of each editor
Brand_counts = df_final["Brand"].value_counts()
# Create the scatter plot
plt.scatter(x=mean_grades, y=Brand_counts)
# Set the x-axis label
plt.xlabel("Mean grade")
# Set the y-axis label
plt.ylabel("Number of occurrences")
# Show the plot
plt.show()


# # Machine learning

# In[149]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression


# In[150]:


scaler = MinMaxScaler(feature_range=(0,1))
df_final["grade"] = scaler.fit_transform(df_final[["grade"]])


# In[151]:


Properties = ['reviews', "Brand", 'gender','recommend_to_friend', 'fragrance_family', 'fragrance_classification', 'volume']
X = df_final[Properties]
y = df_final['grade']
print(X.head())
print(y.head())


# In[152]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Initial amount of samples: #{}".format(X.shape[0]))
print("Number of training samples: #{}".format(X_train.shape[0]))
print("Number of test samples: #{}".format(X_test.shape[0]))

print("X_train:")
print(X_train.head())
print("y_train:")
print(y_train.head())


# In[153]:


X_train_numeric = X_train._get_numeric_data().copy()
X_numeric_cols = X_train_numeric.columns
X_test_numeric = X_test[X_numeric_cols].copy()
X_numeric_cols


# In[154]:


scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(
    X_train_numeric), columns=X_numeric_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.fit_transform(
    X_test_numeric), columns=X_numeric_cols, index=X_test.index)
X_train_scaled.head()


# In[155]:


X.select_dtypes('category')


# In[156]:


X_discrete = X.select_dtypes('category').copy()
X_discrete_encoded = pd.get_dummies(X_discrete, prefix_sep="__")

X_train_discrete_encoded = X_discrete_encoded.loc[X_train.index, :]
X_test_discrete_encoded = X_discrete_encoded.loc[X_test.index, :]
X_train_discrete_encoded.head()


# In[157]:


X_train_processed = pd.concat((X_train_scaled, X_train_discrete_encoded), axis=1)
X_test_processed = pd.concat((X_test_scaled, X_test_discrete_encoded), axis=1)
X_train_processed.head()


# In[158]:


X_train_processed.columns


# In[159]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# # PLOT TWIST

# ## After noticing over fitting, changing the machine learnning from linear regression to classification

# In[160]:


Is_Successful = []
for element in df_final["grade"]:
    if element > 0.8:
        Is_Successful.append(1)
    else:
        Is_Successful.append(0)


# In[161]:


df_final["Is_Successful"] = Is_Successful


# In[162]:


df_final["Is_Successful"] = df_final["Is_Successful"].astype("category")


# # chi2

# In[163]:


from scipy.stats import chi2_contingency
ct1 = pd.crosstab(df_final["Brand"], df_final['Is_Successful'])
chi2_contingency(ct1)


# In[164]:


ct1 = pd.crosstab(df_final["Is_Successful"], df_final['gender'])
ct1.plot(kind='bar', figsize=(5, 2))
chi2_contingency(ct1)


# In[165]:


ct1 = pd.crosstab(df_final["Is_Successful"], df_final['fragrance_classification'])
chi2_contingency(ct1)


# In[166]:


ct1 = pd.crosstab(df_final["Is_Successful"], df_final['fragrance_family'])
chi2_contingency(ct1)


# In[168]:


bins = [0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100, 2250, 2400, 2550, 2700, 2850, 3000,4000]
labels = ['0-150', '150-300', '300-450', '450-600', '600-750', '750-900', '900-1050', '1050-1200', '1200-1350', '1350-1500', '1500-1650', 
          '1650-1800', '1800-1950', '1950-2100', '2100-2250', '2250-2400', '2400-2550', '2550-2700', '2700-2850', '2850-3000', '3000+']
df_final['reviews_discrete'] = pd.cut(df_final['reviews'], bins, labels=labels)


# In[169]:


ct1 = pd.crosstab(df_final["reviews_discrete"], df_final['Is_Successful'])
ct1.plot(kind='bar', figsize=(5, 2))
chi2_contingency(ct1)


# In[170]:


# Use seaborn to create a boxplot of the data

plt.hist(df_final.reviews, bins=50)
plt.xlabel("Interest Rate")
plt.ylabel("Amount of Articles")
# Compute the 25th and 75th percentiles
q25 = df_final["reviews"].quantile(0.25)
q75 = df_final["reviews"].quantile(0.75)

# Calculate the IQR
iqr = q75 - q25

# Identify values that are more than 1.5 times the IQR below the 25th percentile or above the 75th percentile
outliers = df_final[(df_final["reviews"] < q25 - 3*iqr)
               | (df_final["reviews"] > q75 + 3*iqr)]

# Print the indices of the outliers
outliers


# In[171]:


features = ['reviews', "Brand", 'gender', 'recommend_to_friend', 'fragrance_family', 'fragrance_classification', 'volume']
X = df_final[features]
y = df_final['Is_Successful']

print(X.head())
print(y.head())


# In[172]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Initial amount of samples: #{}".format(X.shape[0]))
print("Number of training samples: #{}".format(X_train.shape[0]))
print("Number of test samples: #{}".format(X_test.shape[0]))

print("X_train:")
X_train.head()
print("y_train:")
y_train.head()


# In[173]:


# select numeric features:
X_train_numeric = X_train._get_numeric_data().copy()
X_numeric_cols = X_train_numeric.columns
X_test_numeric = X_test[X_numeric_cols].copy()
X_numeric_cols


# In[174]:


X_test_numeric


# In[175]:


scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_numeric), columns=X_numeric_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test_numeric), columns=X_numeric_cols, index=X_test.index)
X_train_scaled.head()


# In[176]:


X_train_numeric.max().sort_values()


# In[177]:


X.select_dtypes('category')


# In[178]:


X_discrete = X.select_dtypes('category').copy()
X_discrete_encoded = pd.get_dummies(X_discrete, prefix_sep="__")
X_train_discrete_encoded = X_discrete_encoded.loc[X_train.index, :]
X_test_discrete_encoded = X_discrete_encoded.loc[X_test.index, :]
X_train_discrete_encoded.head()


# In[179]:


X_train_processed = pd.concat((X_train_scaled, X_train_discrete_encoded), axis=1)
X_test_processed = pd.concat((X_test_scaled, X_test_discrete_encoded), axis=1)
X_train_processed.head()


# In[180]:


X_train_processed.columns


# # Classification Models

# ### Logistic Regression Best f1 Score : 0.88

# In[207]:


def describe_model(model, X_test, y_test):

    print("Predicting...")
    print("\tdisplaying information re: the 'classification' model ...\n" )
    y_pred = model.predict(X_test)
    print("Model Accuracy: ", model.score(X_test, y_test))
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(conf_mat)

    print("\nClassification Report: \n")
    print(classification_report(y_true=y_test, y_pred=y_pred))


# In[208]:



logistic_reg = LogisticRegression(class_weight='balanced')
logistic_reg.fit(X_train_processed, y_train)

describe_model(logistic_reg, X_test_processed, y_test)


# ## Neural NetWork Best f1 Score : 0.87

# In[209]:


neural_network = MLPClassifier(random_state=1, max_iter=370, hidden_layer_sizes=400,activation="relu").fit(X_train_processed, y_train)
describe_model(neural_network, X_test_processed, y_test)


# ## Decision Tree Best f1 Score : 0.89

# In[184]:


des_tree = tree.DecisionTreeClassifier()
params = {"max_depth": [2, 3, 4, 5, 6, 7],
          "min_samples_split": [5, 10, 15, 20, 25, 30]}
clfCV = GridSearchCV(des_tree, params, cv=10)
clfCV.fit(X_train_processed, y_train)
print(f"best params are:{clfCV.best_params_}")
print(f"best score are:{clfCV.best_score_}")


# In[185]:


des_tree = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=15)
des_tree.fit(X_train_processed, y_train)
describe_model(des_tree, X_test_processed, y_test)


# # Random Foreset Best f1 Score : 0.84

# In[186]:


rand_forest = RandomForestClassifier()
params = {"max_depth": [2, 3, 4, 5, 6, 7],"min_samples_split": [5, 10, 15, 20, 25, 30]}
clfCV = GridSearchCV(rand_forest, params, cv=10)
clfCV.fit(X_train_processed, y_train)
print(f"best params are:{clfCV.best_params_}")
print(f"best score are:{clfCV.best_score_}")


# In[189]:


rand_forest = RandomForestClassifier(max_depth=2, min_samples_split=5)
rand_forest.fit(X_train_processed, y_train)
describe_model(rand_forest, X_test_processed, y_test)


# ## KNN Best f1 Score : 0.89

# In[190]:


knn = KNeighborsClassifier()
params = {"n_neighbors": [3, 5, 7, 9, 11, 13,15,17,19,21,23]}
# Create a k-NN classifier with 3 neighbors
clfCV = GridSearchCV(knn, params, cv=10)
clfCV.fit(X_train_processed, y_train)
# Fit the classifier to the training data
print(f"best params are:{clfCV.best_params_}")
print(f"best score are:{clfCV.best_score_}")


# In[191]:


knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train_processed, y_train)
describe_model(knn, X_test_processed, y_test)


# # In conclusion:

# # The Logistic Regression gives us the highest score among the rest  (0.88) 
