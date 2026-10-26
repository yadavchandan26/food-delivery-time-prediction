import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



# loading the data
def load_data(data_path):
    data=pd.read_csv(data_path)
    return data

#filling NaN
def preprocessing(data):
    numeric_col=data.select_dtypes(include='number').columns.tolist()
    cat_col=data.select_dtypes(include='object').columns.tolist()

    num_imp=SimpleImputer(strategy='median')
    cat_col=SimpleImputer(strategy='most_frequent')

    data[numeric_col]=num_imp.fit_transform(data[numeric_col])
    data[cat_col]=cat_col.fit_transform(data[cat_col])

    return data


#lets get our X and Y
def split(data):
    x=data.drop(['Order_ID','Delivery_Time_min'],axis=1)
    x=pd.get_dummies(x,columns=['Traffic_Level','Weather','Vehicle_Type','Time_of_Day'],drop_first=True)
    y=data['Delivery_Time_min']

    return x,y

#final function that will call in the main.py
def final_processing(data_path):
    data=load_data(data_path)
    data=preprocessing(data)
    x,y=split(data)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=42)
    scale=StandardScaler()
    x_train_scale=scale.fit_transform(x_train)
    x_test_scale=scale.transform(x_test)

    return x_train_scale,x_test_scale,y_train,y_test