

def search(count,list):
    print(count,list)
    if count ==0:
        return list+["lift"]
    else:
        return search(count-1,list +[count])

print(search(10,[]))