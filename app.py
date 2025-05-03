import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd

#Ste the page config
st.set_page_config(page_title="K-Means Clustering", layout="centered")

#Load model 
with open('kmeans_model.pkl','rb') as f:
    loaded_model = pickle.load(f)
    
#Set title 
st.title("K-Means Clustering Visualizer Customer Segmentation")

#Load Dataset
df = pd.read_csv('marketing_campaign.csv')

# Define feature names
feature_names = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 
                'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
                'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']

# Get the feature columns for clustering
X = df[feature_names].values

# Add feature selection dropdowns
x_axis = st.selectbox('Select X-axis feature', feature_names)
y_axis = st.selectbox('Select Y-axis feature', feature_names)

# Get feature indices
x_idx = feature_names.index(x_axis)
y_idx = feature_names.index(y_axis)

#Predict using the loaded model
y_kmeans = loaded_model.predict(X)

#plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, x_idx], X[:, y_idx], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, x_idx], loaded_model.cluster_centers_[:, y_idx], s=300, c='red', label='Centroids')
ax.set_title('k-Means Clustering')
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.legend()
st.pyplot(fig)
