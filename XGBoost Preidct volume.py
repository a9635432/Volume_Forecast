import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import io 

sns.set(style="darkgrid")

import os
for dirname,_, filenames in os.walk("XXX\\predict our volume"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("D:\\predict our volume\\AEtrain.csv", encoding="cp1252", dtype={'HAWB': "string"})
test = pd.read_csv("D:\\predict our volume\\AEtest.csv", encoding="cp1252")
DestRegion = pd.read_csv("D:\\predict our volume\\DestRegion.csv", encoding="cp1252")


'''
remove the outliers
'''

flierprops = dict(marker="o", markerfacecolor = "purple", markersize=6, linestyle ="none", markeredgecolor = "black")

#Freight_Cost outlier
plt.figure(figsize=(10,4))
plt.xlim(train.Freight_Cost.min(), train.Freight_Cost.max()*1.1)
sns.boxplot(x=train.Freight_Cost, flierprops = flierprops)

#Freight_Revenue outlier
plt.figure(figsize = (10,4))
plt.xlim(train.Freight_Revenue.min(), train.Freight_Revenue.max()*1.1)
sns.boxplot(x=train.Freight_Revenue, flierprops = flierprops)

#AWT outlier
plt.figure(figsize = (10,4))
plt.xlim(train.AWT.min(), train.AWT.max()*1.1)
sns.boxplot(x=train.AWT, flierprops = flierprops)

#CWT outlier
plt.figure(figsize = (10,4))
plt.xlim(train.CWT.min(), train.AWT.max()*1.1)
sns.boxplot(x=train.CWT, flierprops = flierprops)


#plt.show()


#remove outliers
train = train[(train.Freight_Cost < 300000) & (train.Freight_Revenue <300000) &(train.AWT < 60000) & (train.CWT < 60000)]

import re
#replace special characters and turn to lower case
train["Shipper"] = train.Shipper.str.replace("[^A-Za-z0-9]+", " ").str.lower()
train["Consignee"] = train.Consignee.str.replace("[^A-Za-z0-9]+", " ").str.lower()


#Fill NA with str(0)
train = train.fillna("0")



from sklearn.preprocessing import LabelEncoder



DestRegion["HouseDestRegion"] = LabelEncoder().fit_transform(DestRegion.HouseDestRegion)
DestRegion["HouseDestRegion"] = DestRegion["HouseDestRegion"].astype(np.int16)



#Making the shipper / Consignee / DestGateway / Airline / Lane / MAWB/ HAWB / HouseDestRegion / TERMS / Lane to numeric form 


train["Airline"] = LabelEncoder().fit_transform(train.Airline)
train["Shipper"] = LabelEncoder().fit_transform(train.Shipper)
train["MAWB"] = LabelEncoder().fit_transform(train.MAWB)
train["HAWB"] = LabelEncoder().fit_transform(train.HAWB)
train["Consignee"] = LabelEncoder().fit_transform(train.Consignee)
train["TERMS"] = LabelEncoder().fit_transform(train.TERMS)



#Change dtypes 
train["date_block_num"] = train["date_block_num"].astype(np.int8)
train["Airline"] = train["Airline"].astype(np.int8)
train["MAWB"] = train["MAWB"].astype(np.int32)
train["HAWB"] = train["HAWB"].astype(np.int32)
train["Shipper"] = train["Shipper"].astype(np.int16)
train["Consignee"] = train["Consignee"].astype(np.int16)
train["TERMS"] = train["TERMS"].astype(np.int8)
train["DestGatewayID"] = train["DestGatewayID"].astype(np.int16)
train["OriginGatewayID"] = train["OriginGatewayID"].astype(np.int8)
train["AWT"] = train["AWT"].astype(np.float32)
train["CWT"] = train["CWT"].astype(np.float32)
train["Freight_Cost"] = train["Freight_Cost"].astype(np.float32)
train["Total_Cost"] = train["Total_Cost"].astype(np.float32)
train["Freight_Revenue"] = train["Freight_Revenue"].astype(np.float32)
train["Total_Rev"] = train["Total_Rev"].astype(np.float32)



'''
preprocessing
--> Create a matrix df with every combination of month, Origin and Dest etc.
'''

from itertools import product
import time
ts = time.time()


matrix=  []

cols = ["date_block_num","OriginGatewayID", "DestGatewayID"  ]


#return a matrix(list) of product(cols)
for i in range(78): #we have 0-77 date_block_num
    sales = train[train.date_block_num ==i] #return all data in sepeate daye_block_num
    matrix.append(np.array(list(product([i],sales.OriginGatewayID.unique(),
                                        sales.DestGatewayID.unique())),dtype=np.int16))
    

#stacks the array of 0-77 date_block_num
matrix = pd.DataFrame(np.vstack(matrix),columns = cols)
matrix.sort_values(cols,inplace = True)
time.time()- ts
    

ts = time.time()

#Dest on that month, on that Origin
group = train.groupby(["date_block_num", "OriginGatewayID", "DestGatewayID"]).agg({"CWT":["sum"]})
group.columns = ["CWT_month"]
group.reset_index(inplace = True)

#left join group to matrix
matrix = pd.merge(matrix, group, on =cols, how="left")
matrix["CWT_month"] = matrix["CWT_month"].fillna(0).astype(np.float32)
matrix.fillna(0, inplace = True)
time.time()-ts


'''
For a test set in month 78
'''
test["date_block_num"] = 78
test["date_block_num"] = test["date_block_num"].astype(np.int8)
test["OriginGatewayID"] = test.OriginGatewayID.astype(np.int8)
test["DestGatewayID"] = test.DestGatewayID.astype(np.int16)




'''
Concatenate matrix and test sets
'''

ts = time.time()

matrix = pd.concat([matrix, test.drop(["OriginGateway", "DestGateway"], axis=1)],
                   ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace = True)
time.time() - ts


'''
Add HouseDestRegion and Lanes data onto matrix df
'''

ts - time.time()
matrix = pd.merge(matrix, DestRegion, on = ["DestGatewayID"], how ="left")

#matrix = pd.merge(matrix, Lanes, on = ["OriginGatewayID", "DestGatewayID","HouseDestRegion"], how ="left")

matrix.fillna(0, inplace = True)
matrix["HouseDestRegion"] = matrix["HouseDestRegion"].astype(np.int16)
#matrix["Lane"] = matrix["Lane"].astype(np.int16)


time.time() - ts




'''
Done with data cleaning
'''

'''
Feature Engineering
Add lag feature to matrix df
'''


#def lag feature function

def lag_feature(df,lags,cols):
    for col in cols:
        print(col)
        tmp = df[["date_block_num","OriginGatewayID", "DestGatewayID", col]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ["date_block_num", "OriginGatewayID", "DestGatewayID",
                             col + "_lag_" + str(i)]
            shifted.date_block_num = shifted.date_block_num +i
            df = pd.merge(df, shifted, on=["date_block_num",
                                           "OriginGatewayID","DestGatewayID"], how ="left")
        return df



#Add CWT_month lag features
ts = time.time()
matrix = lag_feature(matrix, [1,2,3], ["CWT_month"])
time.time() - ts


#Add the previous month's average CWT
#This is E.g. At month 0, how much CWT on average 


ts = time.time()
group = matrix.groupby(["date_block_num"]).agg({"CWT_month" : ["mean"]})
group.columns = ["date_avg_CWT"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix , group, on=["date_block_num"] , how = "left")
matrix.date_avg_CWT = matrix["date_avg_CWT"].astype(np.float32)
matrix=lag_feature(matrix, [1], ["date_avg_CWT"])
matrix.drop(["date_avg_CWT"], axis=1,inplace = True)
time.time()-ts


'''
#Add lag values of CWT_month for month / DestGatewayID
#This is E.g. At month 0, how many tons for each Destination (average)
'''

ts= time.time()
group = matrix.groupby(["date_block_num", "DestGatewayID"]).agg({"CWT_month" : ["mean"]})
group.columns = ["date_dest_avg_CWT"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on = ["date_block_num", "DestGatewayID"], how = "left")
matrix.date_dest_avg_CWT = matrix["date_dest_avg_CWT"].astype(np.float32)
matrix = lag_feature(matrix, [1,2,3], ["date_dest_avg_CWT"])
matrix.drop(["date_dest_avg_CWT"] ,axis = 1, inplace = True)
time.time() - ts



#Add lag values for CWT_month for every OriginID combination
#CWT  per month for each Origin (average)


ts = time.time()
group = matrix.groupby(["date_block_num", "OriginGatewayID"]).agg({"CWT_month": ["mean"]})
group.columns = ["date_ori_avg_CWT"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on = ["date_block_num", "OriginGatewayID"], how = "left")
matrix.date_ori_avg_CWT = matrix["date_ori_avg_CWT"].astype(np.float32)
matrix = lag_feature(matrix, [1,2,3], ["date_ori_avg_CWT"])
matrix.drop(["date_ori_avg_CWT"], axis =1, inplace = True)
time.time()-ts


#Add lag values for CWT_month for month/Ori/Dest
#CWT  per month for each tradelane (average)

ts= time.time()
group = matrix.groupby(["date_block_num", "OriginGatewayID", "DestGatewayID"]).agg({"CWT_month" : ["mean"]})
group.columns  = ["date_ori_dest_avg_CWT"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on = ["date_block_num", "OriginGatewayID", "DestGatewayID"], how = "left")
matrix.date_ori_dest_avg_CWT = matrix["date_ori_dest_avg_CWT"].astype(np.float32)
matrix = lag_feature(matrix, [1,2,3], ["date_ori_dest_avg_CWT"])
matrix.drop(["date_ori_dest_avg_CWT"], axis = 1, inplace = True)
time.time()-ts


#Add lag values for CWT_month/origin/House_dest_region
#each CWT_month in each Origin for each region (average)


ts = time.time()
group = matrix.groupby(["date_block_num", "OriginGatewayID", "HouseDestRegion"]).agg({"CWT_month":["mean"]})
group.columns = ["date_ori_destregion_avg_CWT"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on = ["date_block_num", "OriginGatewayID", "HouseDestRegion"], how = "left")
matrix.date_ori_destregion_avg_CWT = matrix["date_ori_destregion_avg_CWT"].astype(np.float32)
matrix = lag_feature(matrix, [1], ["date_ori_destregion_avg_CWT"])
matrix.drop(["date_ori_destregion_avg_CWT"], axis=1, inplace = True)



#Add lag values for CWT_month for month / House_dest_region
#CWT for each month in each Region (average)

ts = time.time()
group = matrix.groupby(["date_block_num", "HouseDestRegion"]).agg({"CWT_month":["mean"]})
group.columns = ["date_destregion_avg_CWT"]
group.reset_index(inplace = True)

matrix = pd.merge(matrix, group, on =["date_block_num", "HouseDestRegion"], how = "left")
matrix.date_destregion_avg_CWT = matrix["date_destregion_avg_CWT"].astype(np.float32)
matrix = lag_feature(matrix, [1], ["date_destregion_avg_CWT"])
matrix.drop(["date_destregion_avg_CWT"], axis =1, inplace = True)
time.time()-ts




'''
Add average cost price onto matrix df
Add lag values of item cost per month
Add delta price values - how current month average relates to global average
'''

#cost for each dest (avg)
ts = time.time()
group = train.groupby(["DestGatewayID"]).agg({"Total_Cost":["mean"]})
group.columns = ["dest_avg_cost"]
group.reset_index(inplace = True)

matrix = matrix.merge(group, on =["DestGatewayID"], how = "left")
matrix["dest_avg_cost"] = matrix.dest_avg_cost.astype(np.float32)


#The cost for each dest in each month (avg)
ts = time.time()
group = train.groupby(["date_block_num", "DestGatewayID"]).agg({"Total_Cost":["mean"]})
group.columns = ["date_dest_avg_cost"]
group.reset_index(inplace = True)

matrix = matrix.merge(group, on =["date_block_num","DestGatewayID"], how = "left")
matrix["date_dest_avg_cost"] = matrix.date_dest_avg_cost.astype(np.float32)
lags = [1,2,3]

matrix = lag_feature(matrix, lags, ["date_dest_avg_cost"])
for i in lags:
    matrix["delta_cost_lag_" +str(i)] = (matrix["date_dest_avg_cost_lag_" +str(i)]-
                                         matrix["dest_avg_cost"]) / matrix["dest_avg_cost"]
    
                                                
                                                
#if "delta_cost_lag_1" is 0 --> loop "delta_cost_lag_2", if "delta_cost_lag_2" is 0 too -->"delta_cost_lag_3"

def select_trends(row):
    for i in lags:
        if row["delta_cost_lag_" + str(i)]:
            return row["delta_cost_lag_" + str(i)]
        return 0

matrix["delta_cost_lag"] = matrix.apply(select_trends, axis=1)
matrix["delta_cost_lag"] = matrix.delta_cost_lag.astype(np.float32)
matrix["delta_cost_lag"].fillna(0, inplace = True)


#Drop all only has "delta_cost_lag" left
features_to_drop = ["dest_avg_cost", "date_dest_avg_cost"]

for i in lags:
    features_to_drop.append("date_dest_avg_cost_lag_" + str(i))
    features_to_drop.append("delta_cost_lag_"+str(i))
matrix.drop(features_to_drop, axis=1, inplace =True)
time.time()-ts


'''
Add  total revenue per Origin to matrix df
Add lag values of revenue per month
Add delta revenue values - how current month revenue relates to global average
'''


ts= time.time()

#Total revenue per Origin per month 
group = train.groupby(["date_block_num", "OriginGatewayID"]).agg({"Total_Rev":["sum"]})
group.columns = ["date_Ori_revenue"]
group.reset_index(inplace = True)

matrix = matrix.merge(group, on = ["date_block_num", "OriginGatewayID"], how = "left")
matrix["date_Ori_revenue"] = matrix["date_Ori_revenue"].astype(np.float32)

#average revenue per Origin 
group = group.groupby(["OriginGatewayID"]).agg({"date_block_num":["mean"]})
group.columns = ["Ori_avg_revenue"]
group.reset_index(inplace= True)



matrix = matrix.merge(group, on = ["OriginGatewayID"], how = "left")
matrix["Ori_avg_revenue"] = matrix.Ori_avg_revenue.astype(np.float32)

matrix["delta_revenue"]  = (matrix["date_Ori_revenue"]
                            - matrix["Ori_avg_revenue"]) / matrix["Ori_avg_revenue"]
matrix["delta_revenue"] = matrix["delta_revenue"].astype(np.float32)

matrix = lag_feature(matrix, [1], ["delta_revenue"])
matrix["delta_revenue_lag_1"] = matrix["delta_revenue_lag_1"].astype(np.float32)
matrix.drop(["date_Ori_revenue", "Ori_avg_revenue", "delta_revenue"], axis = 1, inplace = True)
time.time() -ts


'''
Add month and number of days in each month to matrix df
'''

matrix["month"] = matrix["date_block_num"] %12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix["days"] = matrix["month"].map(days).astype(np.int8)


'''
Add the month of each shop and item first sale
'''


#how long the Dest has been trade for each Ori
ts= time.time()
matrix["dest_ori_first_trade"]  = matrix["date_block_num"] - matrix.groupby(["DestGatewayID", "OriginGatewayID"])["date_block_num"].transform("min")
time.time() - ts




#how long the Dest has been sold
ts= time.time()
matrix["dest_first_trade"] = matrix["date_block_num"] - matrix.groupby(["DestGatewayID"])["date_block_num"].transform("min")
time.time() - ts


#how many HAWB  per trade-lane

ts = time.time()
group = train.groupby(["OriginGatewayID", "DestGatewayID"]).agg({"HAWB":["count"]})
group.columns = ["Ori_Dest_HAWBcount"]
group.reset_index(inplace = True)
group.fillna(0, inplace = True)

matrix = matrix.merge(group, on =["OriginGatewayID", "DestGatewayID"], how = "left")
matrix.fillna(0, inplace = True)
matrix["Ori_Dest_HAWBcount"] = matrix.Ori_Dest_HAWBcount.astype(np.int16)



#how many HAWB per month per trade-lane

ts = time.time()
group = train.groupby(["date_block_num","OriginGatewayID", "DestGatewayID"]).agg({"HAWB":["count"]})
group.columns = ["date_Ori_Dest_HAWBcount"]
group.reset_index(inplace = True)
group.fillna(0, inplace = True)


matrix = matrix.merge(group, on =["date_block_num", "OriginGatewayID", "DestGatewayID"], how = "left")
matrix.fillna(0, inplace = True)
matrix["date_Ori_Dest_HAWBcount"] = matrix.date_Ori_Dest_HAWBcount.astype(np.int16)






'''
Delete first 3 months from matrix. They don't have lag values
'''

ts = time.time()
matrix= matrix[matrix["date_block_num"] > 3]
time.time() - ts


'''
End of Feature Engineering
'''



'''
Modelling
'''

import gc
import pickle
from xgboost import XGBRegressor
from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = 12,4



data = matrix.copy()
del matrix
gc.collect()

#Use month 78th --> 0-77, 77is 78th as validation for training

X_train = data[data.date_block_num < 66].drop(["CWT_month"], axis=1)
Y_train = data[data.date_block_num < 66]["CWT_month"]

X_valid = data[data.date_block_num.isin([66,67,68,69,70,71,72,73,74,75,76,77])].drop(["CWT_month"], axis=1)
Y_valid = data[data.date_block_num.isin([66,67,68,69,70,71,72,73,74,75,76,77])]["CWT_month"]

X_test = data[data.date_block_num ==78].drop(["CWT_month"], axis=1)


del data
gc.collect()
ts = time.time()

model = XGBRegressor(
    max_depth = 10,
    n_estimators =1000,
    min_child_weight = 0.5,
    colsample_bytree=0.8,
    subsample = 0.8,
    eta=0.1,
#   tree_method = "gpu_hist"
    seed =42)

model.fit(
    X_train, Y_train,
    eval_metric = "rmse",
    eval_set = [(X_train, Y_train), (X_valid, Y_valid)],
    verbose = True,
    early_stopping_rounds= 20)

time.time() -ts










