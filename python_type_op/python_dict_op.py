"""
python字典
是一个无序、以键值对存储的数据类型，数据关联性强、唯一一个映射数据类型。
键：必须是可哈希（不可变的数据类型：字符串、数字、元组、bool）值，并且是唯一的
"""
# 1.创建字典
dict={}
print(type(dict))
# 2.传入关键字
dict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
# 3.获取数据
print ("dict['Name']: ", dict['Name'])
print ("dict['Age']: ", dict['Age'])
# 4.修改字典
dict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
dict['Age'] = 8
print ("dict['Age']: ", dict['Age'])
# 5.删除字典元素
dict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
del dict['Name'] # 删除键 'Name'
# print(dict)  {'Age': 7, 'Class': 'First'}

del dict # 删除字典
# print("after del",dict)  数据为空