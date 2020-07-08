##SECOND LIBRARY FOR MACHINE LEARNING

##PANDAS
import pandas as pd
#base data structure is DICTIONARY

#creating data frame
d={"Id":[1,2,3,4],
   "Name":["raj", "aman", "anuj", "bharat"],
   "Education":["pg","ug","phd","ug"]}

data=pd.DataFrame(d)
print(data)
print(type(data))

####
#READING EXCEL FILE
import pandas as pd
data_info = pd.read_excel(r"C:\Users\DELL\Desktop\feb5.xlsx", sheet_name="info")
data_info

data_mobile=pd.read_excel(r"C:\Users\DELL\Desktop\feb5.xlsx",sheet_name='mobile')
data_mobile


##CONVERTING DataFrame TO CSV
d={"Id":[1,2,3,4],
   "Name":["raj", "aman", "anuj", "bharat"],
   "Education":["pg","ug","phd","ug"]}

data=pd.DataFrame(d)
print(data)
data.to_csv("C:\Users\DELL\Desktop\feb6.csv")

##analysis of data
import pandas as pd
data=pd.read_csv(r"H:\forestfires.csv")
data.head() #return top 5 rows
data.head(15)
data.tail() #return last 5 rows
data.info() #return information about the headers
data.describe() #return only numeric data
#missing data
data.isnull()
data.isnull().sum()


###
var=data["FFMC"]
data

##MISSING VALUE FILL
import matplotlib.pyplot as plt
import matplotlib as pl
data["FFMC"].plot.hist() #mean value
plt.hist(var)
data["FFMC"].fillna(data["FFMC"].mean(), inplace=True)
data.isnull().sum()

data["temp"].plot.hist() #median value
data["temp"].fillna(data["temp"].median(),inplace=True)
data.isnull().sum()

#EXTRACTING SPECIFIC ROW AND COL
d={"Id":[1,2,3,4],
   "Name":["raj", "aman", "anuj", "bharat"],
   "Education":["pg","ug","phd","ug"]}

data=pd.DataFrame(d)
print(data.iloc[0:,0:2])
data.loc[0:, ["Id"]] #to extract a particular column with known name


#DUMMY VARIABLE
var=pd.get_dummies(data["Education"])
var
#drop to make rule valid of number of dummy
var.pop("phd")
var
#####
var=pd.get_dummies(data["Education"], drop_first=True)
var



##MERGE in pandas
import pandas as pd
d1={"Id":[1,2,3,4],
   "Name":["raj", "aman", "anuj", "bharat"],
   "Education":["pg","ug","phd","ug"]}

df1=pd.DataFrame(d1)

d2={"Id": [1,2,3,4],
    "City":['alwar', 'jaipur', 'chandigarh', 'amritsar']}
df2=pd.DataFrame(d2)

#using merge on='key'
final = pd.merge(df1, df2, on='Id')
print(final)

#using how='inner'
final=pd.merge(df1, df2, on='Id', how='inner')
print(final)

##concatinate
data_final=pd.concat((df1, df2), axis=1) ###CONCAT BY ROW
print(data_final)
data_fi=pd.concat((df1, df2), axis=0) ###CONCAT BY COLUMN
print(data_fi)

