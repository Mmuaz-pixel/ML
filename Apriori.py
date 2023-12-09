import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv('online_retail.csv')
data = data.dropna()
# print(data.info())

basket = (data[data['Country']=='France']
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))


# print(basket.info())

def encode(x): 
    if(x<=0):
        return 0
    elif(x>0):
        return 1
    
basketsets = basket.map(encode)
# print(basketsets)
frequentItems = apriori(basketsets, min_support=0.07, use_colnames=True)
rules = association_rules(frequentItems, metric='lift', min_threshold=1)
print(rules.head())