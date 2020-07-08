##MACHINE LEARNING LIBRARY MATPLOTLIB
#matplotlib.pylot
#all above are used for visualization

import matplotlib as pl
import matplotlib.pyplot as plt

##Data
x=[4,3,2,5]
y=[3,1,2,5]
plt.plot(x,y,color='RED') #default color BLUE
plt.show()

##Data
x=[1,2,3,4,5]
y=[1,2,3,4,5]
plt.plot(x,y,color='magenta') #default color BLUE
plt.show()

##BAR GRAPH
##Data
x=[1,2,3,4,5]
y=[1,2,3,4,5]
plt.bar(x,y,color='magenta') #default color BLUE
plt.show()


##Histogram
sal=[100,200,175,145,1223, 112,23]
plt.hist(sal)
plt.show()

plt.hist(sal, orientation="horizontal", color='cyan', width=0.1)
plt.show()

#Scatter PLT
x=[1,2,3,10,5]
y=[2,9,3,4,5]
plt.scatter(x, y, color='magenta')
plt.show()

plt.scatter(x, y, color='magenta')
plt.xlabel("age")
plt.ylabel("salary")
plt.title("AGE vs SALARY")
plt.show()


############
x=[1,4,6,8]
y=[1,4,4,1]
s=[1,2,6,7]
r=[8,2,2,7]
plt.plot(x,y,color='magenta') #default color BLUE
plt.plot(s,r,color='yellow')
plt.title("BAKWAS")
plt.show()

##CUSTOMIZE
p=[1,2,3,4]
q=[1,2,2,1]
plt.plot(x,y,color='cyan', linestyle='dashed')
plt.show()

###########
#Marker='o' to bold the vertex
plt.plot(x,y,color='cyan', linestyle='dashed', 
         marker='o',markerfacecolor='yellow', linewidth=3.5,
        markersize=10)
plt.show()


#############

#SCATTER plot using online retail 2 file
import pandas as pd
data = pd.read_excel(r"C:\Users\DELL\Downloads\online_retail_II.xlsx", sheet_name="Year 2010-2011")
plt.scatter(data['Quantity'],data['Price'],color='magenta')
plt.xlabel('Quantities')
plt.ylabel('Prices')
plt.title('Quantities vs Prices')
plt.show()

###############

#BAR plot customize

t=['First', 'Two', 'THree','Foue']
x=[5,1,6,7]
y=[14,42,6,23]
plt.bar(x,y, tick_label=t, width=0.7, color=['red', 'blue'])
plt.xlabel('X-axis', fontdict={'family':'comic sans',
                             'color':'cyan',
                             'weight':'bold',
                             'size':18})
plt.ylabel('Y-axis', fontdict={'family':'candara',
                             'color':'magenta',
                             'weight':'bold',
                             'size':18})
plt.title('X vs Y', fontdict={'family':'roboto',
                             'color':'red',
                             'weight':'bold',
                             'size':18})
plt.show()

#################################
#college
s=["books", "fees", "bus"]
p=[4000,12000, 3200]
c=['r', 'y', 'b']
plt.pie(p, labels=s, colors=c, shadow=True, startangle=90, explode=(0.1,0.1,0.1),
        radius=1.5, 
        autopct="%1.1f%%")
plt.show()
