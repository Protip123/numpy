import numpy as np

np_array = np.array([1,2,3,4,5])
print(np_array)
print(type(np_array))#type of array

a = np.array([1,2,3,4,5])
print(a)
print(a.shape)#dimension

b = np.array([(1,2,3,4,5),(6,7,8,9,10)])
print(b)
print(b.shape)

c = np.array([(1,2,3,4,5),(6,7,8,9,10)],dtype=float)#convert float
print(c)

#initial placeholder
x = np.zeros((4,5))#array with 0
print(x)
y = np.full((4,5),7)#array with 7
print(y)
z = np.eye(4)#identity matrix
print(z)

#array with random value
w = np.random.random((3,4))
print(w)
v = np.random.randint(1,10,(5,5))#range 1-10
print(v)

#array with evenly placed value
u = np.linspace(40,50,6)#output 6 value
print(u)
t = np.arange(20,30,3)#output gap 3
print(t)

#list->array
list1 = [1,2,3,4,5]
o = np.asarray(list1)
print(o)
print(type(o))

#analyzing np array
print(v.ndim)#no. of dimension
print(v.size)#no. of element
print(v.dtype)#type of data

#arithmetic operation
n = np.random.randint(10,20,(5,5))
print(n+v)
print(np.add(n,v))
print(n-v)
print(np.subtract(n,v))
print(n*v)
print(np.multiply(n,v))
print(n/v)
print(np.divide(n,v))
print(n%v)
print(np.mod(n,v))
print(n**v)
print(np.power(n,v))
print(1/n)
print(np.reciprocal(n))

#arithmetic functions
print(np.min(n))
print(np.max(n))
print(np.argmin(n))
print(np.argmax(n))
print(np.sqrt(n))
print(np.sin(n))
print(np.cos(n))
print(np.cumsum(n))

#reshape aray
r = n.reshape(-1)
print(r)

#broadcasting array
r = np.array([[1],[2],[3]])
print(r)
s = np.array([1,2,3])
print(s)
print(r+s)

#slicing
s = np.array([1,2,3,4,5,6,7,8,9])
print(s[1:5])
print(s[1:])
print(s[:5])
print(s[::2])#jump by 1
s = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(s[1,1:])

#iteration
s = np.array([1,2,3,4,5,6,7,8,9])
print(s)
for i in s :
    print(i)
s = np.array([[1,2,3,4,5],[6,7,8,9,10]])
for i in s:
    for j in i:
        print(j)
for i in np.nditer(s,flags=['buffered'],op_dtypes=["S"]):
    print(i)

#joining
s = np.array([[1,2],[6,7]])
r = np.array([[4,5],[9,10]])
p = np.concatenate((s,r),axis=1)
print(p)
f = np.stack((s,r),axis =1)
f1 = np.hstack((s,r))#row
f2 = np.vstack((s,r))#colums
f3 = np.dstack((s,r))#heights

#spliting
p1 = np.array_split(s,2)
p2 = np.array_split(s,2,axis=1)
print(p1)
print(p2)

#array function
s = np.array([1,2,3,4])
x = np.where(s == 2)#search
x1 = np.where((s%2) == 0)#search condition
x2 = np.searchsorted(s,5)#search sorted
x3 = np.sort(s)#sort
x4 = [True,False,False,True]#filter
x5 = s[x4]
np.random.shuffle(s)#shuffle
print(s)
print(np.unique(s,return_index=True,return_counts=True))#unique data
print(np.resize(s,(2,2)))#resize 1d to 2d
print(s.flatten())#2d to 1d
print(np.ravel(s))#2d to 1d
print(np.insert(s,(2,3),40))#insert
print(np.insert(s,2,[6,7,8],axis=0))

#matrix
a = np.matrix([[1,2],[4,5]])
b = np.matrix([[1,2],[1,2]])
print(a.dot(b))
print(np.transpose(a))#transpose
print(a.T)
print(np.swapaxes(a,0,1))#swape axis 0 into 1
print(np.linalg.inv(a))#inverse
print(np.linalg.matrix_power(a,2))#power of 2
print(np.linalg.det(a))#determination of matrix