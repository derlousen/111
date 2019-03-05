import shelve

fp = shelve.open('shelve_test.dat')

zhangsan = {'age': 38, 'sex': 'Male', 'address': 'SDIBT'}
fp['zhangsan'] = zhangsan
lisi = {'age': 40, 'sex': 'Male', 'qq': '1234567', 'tel': '7654321'}
fp['lisi'] = lisi

fp.close()


fp = shelve.open('shelve_test.dat')

print(fp['zhangsan']['age'])  # 查看文件内容

print(fp['lisi']['qq'])

# fp.close()


import shelve
with shelve.open('asdf') as b:
    b = {
        'b':{'d':5000},
        'c':{'d':5000}
    }
    if 'b' in b:
        print('yes')
    print (list(b.keys()))
    print(b['c']['d'])