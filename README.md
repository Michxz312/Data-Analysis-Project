```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
```

## EDA and Visualization Project
The dataset is taken from kaggle.com : https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities. 
The dataset chosen is an Airbnb data from european cities. This project looks only at barcelona weekdays and weekends to determine what affects the price of an Airbnb in barcelona. Airbnb is a convenient place for travellers to stay and for residents to earn extra revenue from their property. There are many listings with low and high prices, but which aspect is important to determine the price of an Airbnb?
This project is inspired by Airbnb Data Science project : https://mohamedirfansh.github.io/Airbnb-Data-Science-Project/.


```python
data = pd.concat(map(pd.read_csv, ['barcelona_weekdays.csv', 'barcelona_weekends.csv']), ignore_index=True) # combining the csv
df = data.iloc[: , 1:] # delete the first column
print(df.shape)
df
```

    (2833, 19)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>realSum</th>
      <th>room_type</th>
      <th>room_shared</th>
      <th>room_private</th>
      <th>person_capacity</th>
      <th>host_is_superhost</th>
      <th>multi</th>
      <th>biz</th>
      <th>cleanliness_rating</th>
      <th>guest_satisfaction_overall</th>
      <th>bedrooms</th>
      <th>dist</th>
      <th>metro_dist</th>
      <th>attr_index</th>
      <th>attr_index_norm</th>
      <th>rest_index</th>
      <th>rest_index_norm</th>
      <th>lng</th>
      <th>lat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>474.317499</td>
      <td>Entire home/apt</td>
      <td>False</td>
      <td>False</td>
      <td>4.0</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>10.0</td>
      <td>91.0</td>
      <td>1</td>
      <td>1.111996</td>
      <td>0.630491</td>
      <td>526.469420</td>
      <td>17.942927</td>
      <td>915.587083</td>
      <td>20.154890</td>
      <td>2.17556</td>
      <td>41.39624</td>
    </tr>
    <tr>
      <th>1</th>
      <td>169.897829</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>True</td>
      <td>1</td>
      <td>0</td>
      <td>10.0</td>
      <td>88.0</td>
      <td>1</td>
      <td>1.751839</td>
      <td>0.124017</td>
      <td>320.127526</td>
      <td>10.910462</td>
      <td>794.277350</td>
      <td>17.484489</td>
      <td>2.14906</td>
      <td>41.38714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>161.984779</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>4.0</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>9.0</td>
      <td>88.0</td>
      <td>1</td>
      <td>1.670493</td>
      <td>0.080322</td>
      <td>344.073936</td>
      <td>11.726595</td>
      <td>840.673617</td>
      <td>18.505814</td>
      <td>2.15357</td>
      <td>41.37859</td>
    </tr>
    <tr>
      <th>3</th>
      <td>367.956804</td>
      <td>Entire home/apt</td>
      <td>False</td>
      <td>False</td>
      <td>3.0</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
      <td>10.0</td>
      <td>91.0</td>
      <td>1</td>
      <td>1.475847</td>
      <td>0.093107</td>
      <td>400.057449</td>
      <td>13.634603</td>
      <td>946.589884</td>
      <td>20.837357</td>
      <td>2.16839</td>
      <td>41.37390</td>
    </tr>
    <tr>
      <th>4</th>
      <td>196.895292</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>3.0</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
      <td>9.0</td>
      <td>91.0</td>
      <td>1</td>
      <td>1.855452</td>
      <td>0.272486</td>
      <td>346.042245</td>
      <td>11.793678</td>
      <td>792.296039</td>
      <td>17.440874</td>
      <td>2.15238</td>
      <td>41.37699</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2828</th>
      <td>327.460609</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>True</td>
      <td>0</td>
      <td>0</td>
      <td>9.0</td>
      <td>96.0</td>
      <td>1</td>
      <td>2.608873</td>
      <td>0.932342</td>
      <td>271.642736</td>
      <td>10.488440</td>
      <td>537.512628</td>
      <td>11.807347</td>
      <td>2.19672</td>
      <td>41.39929</td>
    </tr>
    <tr>
      <th>2829</th>
      <td>242.977169</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>9.0</td>
      <td>80.0</td>
      <td>0</td>
      <td>1.859134</td>
      <td>0.555944</td>
      <td>343.045595</td>
      <td>13.245387</td>
      <td>679.561854</td>
      <td>14.927691</td>
      <td>2.19099</td>
      <td>41.39269</td>
    </tr>
    <tr>
      <th>2830</th>
      <td>138.943841</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>96.0</td>
      <td>2</td>
      <td>2.670450</td>
      <td>0.968673</td>
      <td>266.091811</td>
      <td>10.274112</td>
      <td>534.080063</td>
      <td>11.731945</td>
      <td>2.19738</td>
      <td>41.39954</td>
    </tr>
    <tr>
      <th>2831</th>
      <td>185.258454</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>60.0</td>
      <td>1</td>
      <td>2.266090</td>
      <td>0.387429</td>
      <td>290.526738</td>
      <td>11.217573</td>
      <td>627.521382</td>
      <td>13.784536</td>
      <td>2.19679</td>
      <td>41.39029</td>
    </tr>
    <tr>
      <th>2832</th>
      <td>254.614006</td>
      <td>Private room</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>10.0</td>
      <td>100.0</td>
      <td>1</td>
      <td>2.394845</td>
      <td>0.954950</td>
      <td>284.900718</td>
      <td>11.000346</td>
      <td>564.678656</td>
      <td>12.404093</td>
      <td>2.19590</td>
      <td>41.39637</td>
    </tr>
  </tbody>
</table>
<p>2833 rows × 19 columns</p>
</div>




```python
# Look for unique values of the categorical columns
print(df["room_type"].unique())
# renaming biz column to business
df = df.rename(columns={"biz": "business"})
# dropping attr_index, attr_index_norm, rest_index, and rest_index_norm since no information can be found
df = df.drop(['attr_index', 'attr_index_norm', 'rest_index', 'rest_index_norm'], axis=1)
```

    ['Entire home/apt' 'Private room' 'Shared room']
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2833 entries, 0 to 2832
    Data columns (total 15 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   realSum                     2833 non-null   float64
     1   room_type                   2833 non-null   object 
     2   room_shared                 2833 non-null   bool   
     3   room_private                2833 non-null   bool   
     4   person_capacity             2833 non-null   float64
     5   host_is_superhost           2833 non-null   bool   
     6   multi                       2833 non-null   int64  
     7   business                    2833 non-null   int64  
     8   cleanliness_rating          2833 non-null   float64
     9   guest_satisfaction_overall  2833 non-null   float64
     10  bedrooms                    2833 non-null   int64  
     11  dist                        2833 non-null   float64
     12  metro_dist                  2833 non-null   float64
     13  lng                         2833 non-null   float64
     14  lat                         2833 non-null   float64
    dtypes: bool(3), float64(8), int64(3), object(1)
    memory usage: 274.0+ KB
    

### Finding Outliers


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>realSum</th>
      <th>person_capacity</th>
      <th>multi</th>
      <th>business</th>
      <th>cleanliness_rating</th>
      <th>guest_satisfaction_overall</th>
      <th>bedrooms</th>
      <th>dist</th>
      <th>metro_dist</th>
      <th>lng</th>
      <th>lat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2833.000000</td>
      <td>2833.000000</td>
      <td>2833.000000</td>
      <td>2833.000000</td>
      <td>2833.000000</td>
      <td>2833.000000</td>
      <td>2833.000000</td>
      <td>2833.000000</td>
      <td>2833.000000</td>
      <td>2833.000000</td>
      <td>2833.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>293.753706</td>
      <td>2.616661</td>
      <td>0.385104</td>
      <td>0.325450</td>
      <td>9.291564</td>
      <td>91.109072</td>
      <td>1.161313</td>
      <td>2.116982</td>
      <td>0.441248</td>
      <td>2.169379</td>
      <td>41.393495</td>
    </tr>
    <tr>
      <th>std</th>
      <td>355.467888</td>
      <td>1.153124</td>
      <td>0.486706</td>
      <td>0.468625</td>
      <td>1.014577</td>
      <td>8.607153</td>
      <td>0.517108</td>
      <td>1.377859</td>
      <td>0.284540</td>
      <td>0.019545</td>
      <td>0.016138</td>
    </tr>
    <tr>
      <th>min</th>
      <td>69.588289</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.095828</td>
      <td>0.012994</td>
      <td>2.093470</td>
      <td>41.349540</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>160.821095</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>88.000000</td>
      <td>1.000000</td>
      <td>1.070443</td>
      <td>0.256139</td>
      <td>2.156640</td>
      <td>41.381180</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>208.299393</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>93.000000</td>
      <td>1.000000</td>
      <td>1.732003</td>
      <td>0.376994</td>
      <td>2.170960</td>
      <td>41.390390</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>329.787977</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>97.000000</td>
      <td>1.000000</td>
      <td>2.967922</td>
      <td>0.555944</td>
      <td>2.179730</td>
      <td>41.404260</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6943.700980</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>100.000000</td>
      <td>6.000000</td>
      <td>8.443967</td>
      <td>4.039111</td>
      <td>2.225520</td>
      <td>41.463080</td>
    </tr>
  </tbody>
</table>
</div>



- realSum has max of 6943.7, but the mean is only 293.75 with std of 355. The 75 percentile is 329, so any sum bigger than 600 will be dropped.


```python
row = df.loc[df['realSum'] >= 600].index
df = df.drop(row)
print(df.shape)
```

    (2621, 15)
    

## Dataset Explanation
### Columns of the dataset:
- realSum - the total price of the listing (Numeric)
- room_type - the type of room being offered (Categorical : 'Entire home/apt' 'Private room' 'Shared room')
- room_shared - Whether the room is shared or not (Boolean)
- room_private - Whether the room is private or not (Boolean)
- person_capacity - the maximum number of people that can stay in the room (Numeric)
- host_is_superhost - Whether the host is a superhost or not (Boolean)
- multi - whether the listing have multiple rooms (Boolean)
- business - Whether the listing is for business purposes or not (Boolean)
- cleanliness_rating - The cleanliness rating of the listing (Numeric)
- guest_satisfaction_overall - The overall guest satisfaction rating of the listing (Numeric)
- bedrooms - The number of bedrooms in the listing (Numeric)
- dist - The distance from the city centre (Numeric)
- metro_dist - The distance from the nearest metro station (Numeric)
- lng - The longitude of the listing (Numeric)
- lat - The latitude of the listing (Numeric)

### Other info:
There are 2833 rows in the dataset with 15 columns with 212 rows dropped, so there are 2621 rows in the dataset for analysis. There are no null values in the dataset, so imputation is not needed.

### Analyzing the listings based on room types


```python
df["room_type"].value_counts().plot.barh(figsize=(5,2)).set(
    xlabel="count", ylabel="room type", title="room_type histogram"
);
```


    
![png](Price%20of%20Barcelona%20Airbnb_files/Price%20of%20Barcelona%20Airbnb_11_0.png)
    


Most of the listings are private room and it is four times more frequent than the entire home/apt. Shared room is the least of room_type available.


```python
df["person_capacity"].value_counts().plot.bar(figsize=(3,3)).set(
    xlabel="person_capacity", ylabel="count", title="person capacity count histogram"
);
```


    
![png](Price%20of%20Barcelona%20Airbnb_files/Price%20of%20Barcelona%20Airbnb_13_0.png)
    


Two person capacity has the most availability, followed by four person capacity. It is surprising that there are limited choices for more than two people.

### Analyzing the prices for the different room and person capacity.


```python
plt.figure(figsize=(6, 3))
df_pivot = df.pivot_table(values='realSum', index='room_type', columns='person_capacity', aggfunc='mean')
sns.heatmap(df_pivot, annot=True, fmt='.1f', cmap='Blues')
plt.suptitle('Mean Price')
plt.xlabel('person_capacity')
plt.ylabel('room_type')
plt.show()
```


    
![png](Price%20of%20Barcelona%20Airbnb_files/Price%20of%20Barcelona%20Airbnb_16_0.png)
    


It is to be expected that shared room is the cheapest especially if it has a 2 person capacity, and the second cheapest is a shared room with 137.6 with a 4 person capacity. Thus, a shared room is the cheapest. The most expensive is an entire home/apartment for a two person capacity. From the trend of the private room type, more person capacity is more expensive. However, this trend does not apply to the entire home/apartment room type. A two person capacity is more expensive than a six person capacity, but after three person capacity, the more people in the room type, the more expensive it is. This may be because the host of the Airbnb believe that privacy will make people pay more. Then, we can look into customer's satisfaction with room_type and person_capacity.


```python
plt.figure(figsize=(6, 3))
df_pivot = df.pivot_table(values='guest_satisfaction_overall', index='room_type', columns='person_capacity', aggfunc='mean')
sns.heatmap(df_pivot, annot=True, fmt='.1f', cmap='Reds')
plt.suptitle('satisfaction')
plt.xlabel('person_capacity')
plt.ylabel('room_type')
plt.show()
```


    
![png](Price%20of%20Barcelona%20Airbnb_files/Price%20of%20Barcelona%20Airbnb_18_0.png)
    


The previous suggestion that privacy will create more satisfaction doesn't seem to apply. The room types or the person capacity doesn't affect the satisfaction of the guests. There might be other reason that affects the guests's satisfaction more.

### Analyzing the price for geographical location in Barcelona


```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,5))

df.plot(kind="scatter", x="lng", y="lat", c="realSum", cmap=plt.get_cmap("jet"), colorbar=True, alpha=0.3, ax=axes[0])
df.plot(kind="scatter", x="lng", y="lat", c="dist", cmap=plt.get_cmap("jet"), colorbar=True, alpha=0.3, ax=axes[1])
plt.show()
```


    
![png](Price%20of%20Barcelona%20Airbnb_files/Price%20of%20Barcelona%20Airbnb_21_0.png)
    


From the graph above, the further it is from the city center, the cheaper it gets. It is expected that the price of Airbnb is cheaper in places far from the city center because house/apartment prices are more expensive in the city center. The most expensive Airbnb is concentrated in the city center while other less expensive place is still arround the city center. Although we can see some expensive place far from the city center, we can see that the city center are inclined to be expensive.

### Analyzing listing based on other factors


```python
fig = plt.figure(figsize=(10,3)) 

fig_dims = (1, 3)

plt.subplot2grid(fig_dims, (0, 0))
df['host_is_superhost'].value_counts().plot(kind='barh', 
                                     title='superhost count')
plt.subplot2grid(fig_dims, (0, 1))
df['multi'].value_counts().plot(kind='barh', 
                                     title='multi count')
plt.subplot2grid(fig_dims, (0, 2))
df['business'].value_counts().plot(kind='barh', 
                                     title='business count')
```




    <AxesSubplot:title={'center':'business count'}>




    
![png](Price%20of%20Barcelona%20Airbnb_files/Price%20of%20Barcelona%20Airbnb_24_1.png)
    



