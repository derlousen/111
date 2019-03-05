import shelve
with shelve.open('ubuntu_test') as b:
    # b['total']=2
    print(b['total'])
