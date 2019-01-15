def types(a):
    t=[]
    for i in a:
        t.append(type(i))
    print (t)


types(['a',2,4.3])