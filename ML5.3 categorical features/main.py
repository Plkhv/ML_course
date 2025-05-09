import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(way):
    data = pd.read_csv(way)
    return data

def preprocess_customer_support(cs_data):
    cs_data.fillna('Unknown', inplace=True)
    
    cs_data["Customer_Remarks_New"] = (cs_data["Customer Remarks"].str.len() > 3).astype(int)
    cs_data["Is_order"] = cs_data["Order_id"].notna().astype(int)
    
    rare_categories = cs_data["Sub-category"].value_counts()[cs_data["Sub-category"].value_counts() < 5000].index
    cs_data["Sub-category"] = cs_data["Sub-category"].replace(rare_categories, "Other")
    
    cs_data['Agent_case_count'] = cs_data.groupby("Agent_name")["Agent_name"].transform('count')
    
    cs_data.drop(["Unique id", "Order_id", "order_date_time", "Issue_reported at", "Survey_response_Date", "Customer Remarks", "Customer_City", "Product_category", "Item_price", "connected_handling_time"], axis=1, inplace=True)
    
    cs_data = pd.get_dummies(cs_data)
    
    return cs_data

def preprocess_titanic(titanic_data):
    titanic_data.fillna('Unknown', inplace=True)

    le = LabelEncoder()
    titanic_data["Sex"] = le.fit_transform(titanic_data["Sex"])
    titanic_data = pd.get_dummies(titanic_data, columns=["Embarked", "Pclass"])
    
    return titanic_data

def main():
    titanic_data = load_data('data/titanic.csv')
    cs_data = load_data('data/Customer_support_data.csv')
    cs_data = preprocess_customer_support(cs_data)
    titanic_data = preprocess_titanic(titanic_data)
    
    print("Customer Support Data Processed:")
    print(cs_data.head())
    print("\nTitanic Data Processed:")
    print(titanic_data.head())
    
main()