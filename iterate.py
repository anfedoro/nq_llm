def iterate(a):
    while True:
        for i in a:
            yield i

a = [1,2,3,4,5,6,7,8,9,10]

itr = iterate(a)


for j in range(5):
    for i in itr:
        print(j, i)
    
