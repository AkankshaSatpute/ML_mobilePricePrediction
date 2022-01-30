import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix


#importing the dataset
df = pd.read_csv(r"C:\Users\PC Point\Desktop\image\mobile_prices.csv")
print(df)

x = df.iloc[:,0:-1]
y = df.iloc[:,-1]
print(x)
print(y)

#splitting the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("\n\t x_train")
print(x_train)
print("\n\t x_test")
print(x_test)
print("\n\t y_train ")
print(y_train)
print("\n\t y_test")
print(y_test)

#fitting the dataset
classifier = GaussianNB()
classifier.fit(x_train, y_train)

#predictions
y_pred = classifier.predict(x_test)
print("Y_Predict : ",y_pred)



print("\n\t Printing Individual values")
print("\n\t Accuracy : ",metrics.accuracy_score(y_test,y_pred)*100)
print("\n\t Precision : ",metrics.precision_score(y_test,y_pred,average = 'weighted')*100)
print("\n\t Recall : ",metrics.recall_score(y_test, y_pred,average = 'weighted' )*100)


