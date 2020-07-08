
import numpy as np

#creating numpy array using list
dis  = np.array([10,20,30])
tim = np.array([1,22,3])
print(dis//tim)

#creating numpy array as tuple
dis=np.array((10,100,200))
tim = np.array((1,2,3))
print(dis//tim)

#single value in tuple acts as an integer
dis=np.array((1))
print(type(dis))

dis=np.array((1,2,3,4))
print(type(dis))

d=(1,2,3,4)
dis=np.array(d)
print(type(d))
print(type(dis))

#creating nested list array
l=[[1,2],[3,4],[5,6]]
dis=np.array(l)
print(dis)
print(dis[2][1])
#checking the dimension
print(dis.ndim)

#printing row and column
print(dis.shape)

#printing no. of elements in ndarray
print(dis.size)

#printing the data type of the ndarray
print(dis.dtype)

#changing the datat type of the of the ndarray
dis=np.array(l, dtype='float')
print(dis)
print(dis.dtype)

#creating the nested tuple array
t=((1,2),
   (45,32),
   (67,23))
tis=np.array(t)
print(t)
print(tis.ndim)
print(tis.shape) #no. of row and column
print(tis.dtype) #printing the data type of the array


#importing the random funtion
import random 
print(random.random())

#creating the array with random array 
#it returns value between 0 and 1 without including 0 and 1
ar=np.random.random((3,3))
print(ar)
print(ar.dtype)
print(ar.shape)

#creating zero matrix
#writing the no. of rows and columns
a=np.zeros((2,3))
print(a)

#printing the random no. between range
a=np.random.randint(10,100)
print(a)


'''
a=np.random.random((2,3), dtype='int')
print(a)
'''

#look below program to understand arange
t=np.arange(0,25,5)
print(t)
t=np.arange(25,-1,-5)
print(t)


#working of a arange
s=[]
for i in range(0,26,5):
    s.append(i)
t=np.array(s)
print(t)

#reshaping the array
l=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
arr=np.array(l)
#print(arr)
arr=arr.reshape(2,2,3)
print(arr)
print(type(arr))
print(arr.shape)
print(arr.dtype)


#converting into single element
l=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
arr=np.array(l)
f=arr.flatten()
print(f)


#slicing the array
##array_name[start : stop : increament value]
## increament_value == -1 repesents the last value of the array
#for ndarray [row, column]

l=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
arr=np.array(l)
print(arr[0:])
print(arr[::-1])
#reversing the row and column
print(arr[0:,::-1])
print(arr)
print(arr[-1::-1,::-1])
#or
print(arr[len(arr)-1::-1, ::-1])


#BASIC OPERATIONS ON ARRAY
l=[1,2,3,4,5,6]
arr=np.array(l)
d=arr*5 #multiply
print(d)
d=arr/5 #float divison
print(d) 
d=arr**2 #power
print(d)
##working of arr**2
'''
l=[1,2,3,4,5,6]
a=list(map(lambda x:x**2,1))
print(a)
'''

#transposing the element of the array Using T transpose method 
l=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
arr=np.array(l)
d = arr.T
print(arr,"\n")
print(d)


#UNARY OPERATIONS
#MAX AND MIN FUNCTION
l=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
arr=np.array(l)
print(arr.max())
print(arr.min())
#for row axis=1 and colum axis =0
l=[[12,24,83],[41,105,456],[37,83,95],[710,411,812]]
arr=np.array(l)
print(arr.max(axis=1)) #maximum in each row
print(arr.max(axis=0)) #maximum in each colum
print(arr.min(axis=1)) #minimum in each row
print(arr.min(axis=0)) #minimum in each colum

#SUM()
print(arr.sum())
print(arr.sum(axis=1)) #for each row
print(arr.sum(axis=0)) #for each column

#BINARY OPERATION ON ARRAY
a=np.array([[1,2],
            [3,4]])
b=np.array([[10,20],
            [30,40]])
print(a+b)

#UNIVERSAL OPERATION
import math
d=math.sqrt(225)
print(d)
b=np.array([1,2,3,4,5])
b=np.sqrt(b)
print(b)
a=np.array([[1,2],
            [3,4]])
sa=np.sqrt(a)
print(sa)


##WORKING OF SQRT
d=[]
a=[1,2,3,4,5]
for i in a:
    d.append(math.sqrt(i))
arr=np.array(d)
print(arr)

#SORTING OF THE ARRAY
l=[[12,24,83],[41,105,456],[37,83,95],[710,411,812]]
arr=np.array(l)
print(np.sort(arr)) #default sorting according to each row
print(arr[-1::-1,::-1]) ##REVERSE ORDER
print(np.sort(arr, axis=1)) #sorting each row
print(np.sort(arr, axis=0)) #sorting each column

##STACKING
a=np.array([[1,2],
            [3,4]])
b=np.array([[10,20],
            [30,40]])
arr_h=np.hstack((a,b))
print(arr_h)
print(arr_h.shape)

arr_v=np.vstack((a,b))
print(arr_v)
print(arr_v.shape)


##BROADCASTING --> row == column
a=np.array([[1,2],
            [3,4]])
b=np.array([[10,20],
            [30,40]])
print(a*b)

##value error ValueError: operands 
#could not be broadcast together with shapes (2,2) (2,3) 
a=np.array([[1,2],
            [3,4]])
b=np.array([[10,20,1],
            [30,40,2]])
print(a+b)










 






