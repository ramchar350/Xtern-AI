import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('XTern 2024 Artificial Intelegence Data Set.xlsx')

print(data.head())

summary_stats = data.describe()
print(summary_stats)

major_counts = data['Major'].value_counts()
major_order = major_counts.index

year_counts = data['Year'].value_counts()
year_order = year_counts.index

uni_counts = data['University'].value_counts()
uni_order = uni_counts.index

ord_counts = data['Order'].value_counts()
ord_order = ord_counts.index

# Create plots for the different columns to make inferences for the data
plt.figure(figsize=(8, 6))
sns.countplot(data['Major'], order=major_order)
plt.title('Distribution of the Major Column')
plt.xlabel('Count')
plt.ylabel('Count')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data['Year'], order=year_order)
plt.title('Distribution of the Year Column')
plt.xlabel('Count')
plt.ylabel('Count')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data['University'], order=uni_order)
plt.title('Distribution of the University Column')
plt.xlabel('Count')
plt.ylabel('Count')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data['Order'], order=ord_order)
plt.title('Distribution of the Order Column')
plt.xlabel('Count')
plt.ylabel('Count')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['Time'], kde=True)
plt.title('Distribution of the Time Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()