"""
Python类中的__init__() 和 self 的解析
"""


class Person(object):
    def __init__(self, name, lang, website):  #__init__为默认调用的方法   为构造方法
        self.name = name
        self.lang = lang
        self.website = website

        print('self: ', self)
        print('type of self: ', type(self))


'''
未实例化时，运行程序，构造方法没有运行
'''

p = Person('Tim', 'English', 'www.universal.com')

'''实例化后运行的结果
self:  <__main__.Person object at 0x00000000021EAF98>
type of self:  <class '__main__.Person'>
'''