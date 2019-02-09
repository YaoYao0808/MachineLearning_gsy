"""
集合（set）是一个无序的不重复元素序列。

可以使用大括号 { } 或者 set() 函数创建集合，注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典。

创建格式：
"""
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(type(basket))  #<class 'set'>

basket1 = ['apple', 'apple','orange', 'pear', 'banana']
print(type(basket1))  #<class 'list'>

# 综上所述，set && list
"""
set使用花括号{}创建，且里面元素不能重复
list使用方括号[]创建，里面元素可以重复
"""


basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
# print(basket)  {'orange', 'pear', 'banana', 'apple'}  自动去重
print('orange' in basket) # 快速判断元素是否在集合内)  True

# 两个集合间的运算
a = set('abracadabra')
b = set('alacazam')
# print(a)  {'c', 'r', 'a', 'b', 'd'}
print(a-b)  #a - b # 集合a中包含而集合b中不包含的元素  {'b', 'r', 'd'}
# a & b 集合a和b中都包含了的元素

# 集合的基本操作
"""
s.add( x )
s.remove( x )   s.discard( x )元素如果不存在，不会发生错误
s.pop()  随机删除集合的任意一个元素
"""
