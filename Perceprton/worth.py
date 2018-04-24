import numpy as np

a = [10, 'abs', 'dasd']
c = np.array(a)
c = np.repeat(c, 10)
b = 10 * a
print(b)
print(c)

string = ('This is a single long, long string'
          ' written over many lines for convenience'
          ' using implicit concatenation to join each'
          ' piece into a single string without extra'
          ' newlines (unless you add them yourself).')

print (string[0:40])
i = 0
for a in range (1, 3):
    print ('d')
a = [1,2,3,4,5]
print(a[0:])