# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("ulabox_orders_with_categories_partials_2017.csv")

dfp = df[["Drinks%", "Food%","Fresh%","Baby%","Pets%","Beauty%","Home%","Health%"]]

ssd = []
ks = range(1,11)
for k in range(1,11):
    km = KMeans(n_clusters=k)
    km = km.fit(dfp)
    ssd.append(km.inertia_)

kneedle = KneeLocator(ks, ssd, S=1.0, curve="convex", direction="decreasing")
kneedle.plot_knee()
plt.show()
# %%
k = round(kneedle.knee)

print(f"Number of clusters suggested by knee method: {k}")
# %%

kmeans = KMeans(n_clusters=k).fit(df[["Drinks%", "Food%"]])

plt.show()

# %%
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier()
tree.fit(df[["weekday", "Drinks%", "Food%"]], kmeans.labels_)
print(export_text(tree, feature_names=["weakday", "Drinks%", "Food%"]))
# %%
cluster0=df[kmeans.labels_==0]
cluster1=df[kmeans.labels_==1]
cluster2=df[kmeans.labels_==2]
cluster3=df[kmeans.labels_==3]
cluster4=df[kmeans.labels_==4]

# %%
sns.boxplot(data=cluster0, x=df['weekday'], y="Food%")
sns.boxplot(data=cluster0, x=df['weekday'], y="Fresh%")
sns.boxplot(data=cluster0, x=df['weekday'], y="Drinks%")
sns.boxplot(data=cluster0, x=df['weekday'], y="Home%")
sns.boxplot(data=cluster0, x=df['weekday'], y="Beauty%")
sns.boxplot(data=cluster0, x=df['weekday'], y="Health%")
sns.boxplot(data=cluster0, x=df['weekday'], y="Baby%")
sns.boxplot(data=cluster0, x=df['weekday'], y="Pets%")

# %%
