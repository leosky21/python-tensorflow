# -*- coding: utf-8 -*-
## 引入新的包
import turtle
import pickle  # 文件操作
import tensorflow as tf # alias tf
from time import time, localtime # 引入想要的功能
from time import *  # 引入所有功能
## 引入自定义包:  在Mac系统中，下载的python模块会被存储到外部路径site-packages，同样，我们自己建的模块也可以放到这个路径，最后不会影响到自建模块的调用。


"""
If the module doesn't exist, 
    pip install name_module  # for python2
    pip3 install name_module # for python3
    
    pip install name_module=verson_num  # for python install specific versions
    pip3 install name_module=version_num # for python3 specific versions
    
    pip install -U name_module # for update
    pip3 install -U name_module # for update
"""
## 输入
a_input = input('please input a number:')
score = int(input('Please input your score: \n'))

## 输出
print("hello, world")

## 占位符
## %d	整数
## %f	浮点数
## %s	字符串
## %x	十六进制整数
print('Age: %s. Gender: %s' % (25, True))

# format 占位符 {}
print('Hello, {0}, 成绩提升了 {1:.1f}%'.format('小明', 17.125))


### 赋值,运算
### 特殊: 多变量赋值
a, b, c = 1, 2, 3
print(a, b, c)

## 加,减,乘,除,取余,取整, 乘方
## + , -, * , /, %, // , **
print(2**4)


## 编码
ord('中')  # 获取字符的整数表示
chr(12321)  # 获取编码对应的字符

'ABC'.encode('ascii')  # 编码为指定的bytes
'中文'.encode('utf-8')

b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8')  # 读取字节流封装

len('ABC')  # 计算字节长度
len('\xe4\xb8\xad\xe6\x96\x87')

## range使用: 工厂函数
range(1, 10)
stop = 10
range(stop=stop) ## it means range(0,10)
step = 2
range(1, stop=stop, step=step) # it means range from 1 to 10 step by 2

## 条件语句
## python 用缩进表示代码块
age = input("input your age\n")
if int(age) > 18:
    print("already 成年")
elif int(age) > 30:
    print("30 more")
else:
    print('<18')
print('out of if else ')

#  A exercise  #

L = [
    ['Apple', 'Google', 'Microsoft'],
    ['Java', 'Python', 'Ruby', 'PHP'],
    ['Adam', 'Bart', 'Lisa']
]

print(L[0][0])
print(L[1][1])
print(L[2][2])

# 循环语句

for i in range(0, 7):
    print("hello world")

# while a<10 and b<20
t = turtle.Pen()
for i in range(0, 4):
    t.forward(100)
    t.left(90)

t = (1, 2)

## while 打印0-9 的数据
'''
while True:
    print("always print True")
'''
conditions = 0
while conditions < 10:
    print(conditions)
    conditions += 1

## 比较操作
'''
    比较运算符:
        小于（<)
        大于 (>)
        不大于 (<=)
        不小于 (>=)
        等于 (==)
        不等于 (!=)
            会return True 和 False
'''


# 条件判断
# 条件 是 非None,非零, 则是True
# (list、 tuple 、dict 和 set )集合中,如果元素数量为0 ,则是False
age = 20
if age >= 18:
    print("your age is ", age)
    print('adult')
elif age >= 6:
    print('teenager')
else:
    print('kid')

worked = True
isDone = 'Done' if worked else 'not yet'
print(isDone)

# 循环判断
for name in c:
    print(name)


print("range 函数生成整数序列 : ", range(4))

# continue  和  break



# 内置集合: list[], tuple()(用圆括号或者不用括号), dict{}, set([])  ,每个集合都能迭代

## list 列表名 [ 索引地址值 ]
### name[ 1 :5] from 1 ~ 5

### name[ -1: -2 ] 倒数第二个开始,步长为2
### list 可变
c = [1, 2, 3, "张三"]
c.append("在末尾追加")
c.insert(1, "在位置1追加")
c.remove(2) # 移除 第一个值为2 的项
c.index(2)  # 显示第一个出现2 的索引
c[2:]  # 第二位以及之后的所有的项, 也可以理解为地址为2之后的地址为3以及之后的所有的值
c[-2:] # 列表倒数第二位以及,倒数第二位之后(从左向右)的所有项
c[0:3] # 第0位到第2位(第3位之前)所有项的值
#c.sort() # 默认从小到大, c.sort(reverse=True) # 从大到小排序
print(c[1])
print(c[2:4])
print(c[-1:: -2])
print(" list c 的长度 : %s " % len(c))

## 多为列表
a = [1 ,2,3,4,5]     # 一行五列
multi_dim_a = [[1,2,3],
			   [2,3,4],
			   [3,4,5]]  # 三行三列


## tuple 元祖  不可变
classmates = ('Michael', 'Bob', 'Tracy')
classmates02 = 'Michael', 'Bob', 'Tracy'
t = ('a', 'b', ['A', 'B'])
t[2][0] = 'X'
t[2][1] = 'Y'
# t =  ('a', 'b', ['X', 'Y'])


##  dict ,也就是字典 (其实也就是ap),key - value 对应查询
###  key 如果不存在就会报错 , get() 查询,不存在则为none
###  特点:
###      查找和插入的速度极快，不会随着key的增加而变慢；
###      需要占用大量的内存，内存浪费多。

## dict 是非有序的,需要顺序一致的dict, 使用collections模块中的OrderDict对象

d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
print("字典: d.get('xx')  =  d['xx']: ", d.get('Bob'))
# 删除的两种方法
d.pop('Bob')
del d['Tracy']
d[1] = 20  # 添加的一种方法,直接添加

## 字典是无序容器,存储类型多变,
d4 = {'apple': [1,2,3], 'pear':{1:3, 3:'a'}, 'orange':func}
print(d4['pear'][3])    # a

## set : 自带去重属性
## add , remove
s = set([1, 2, 3])
s.add(4)
print("s.add() ", s)
s.remove(4)
print("s.remove :: ", s)
print("s.pop() ", s.pop())
print("after s.pop()::", s)


## 迭代器
## 生成器

# 定一个函数
def hi_name(yourname):
    print(yourname)
def hi(yourname, num):
    print(yourname,num)
## 默认参数
## 函数声明只需要在需要默认参数的地方用 = 号给定即可, 但是要注意所有的默认参数都不能出现在非默认参数的前面
def function_name(price, color='red', brand='carmy', is_second_hand=True):
    print('price', price,
          'color', color,
          'brand', brand,
          'is_second_hand', is_second_hand, )
hi_name("这是自定义函数的定义输出")

## 自调用
'''
    如果执行该脚本的时候，该 if 判断语句将会是 True,那么内部的代码将会执行。
    如果外部调用该脚本，if 判断语句则为 False,内部代码将不会执行。 
'''
if __name__ == '__main__':
    #code_here
    print("测试使用的")

## 可变参数
def report(name, *grades):
    total_grade = 0
    for grade in grades:
        total_grade += grade
    print(name, 'total grade is ', total_grade)
report('wang', 12,14,16,20)

## 关键字参数
## universal_func(*args, **kw) ==> 可以代表任何函数
def portrait(name, **kw):
    print('name is', name)
    for k, v in kw.items():
        print(k, v)
print(portrait('Mike', age=24, country='China', education='bachelor'))

## 全局变量
global_param = 100
def for_local():
    local_parm = 20
    print("value of local_parm is %d" % local_parm)
    print("value of global_parm is %d" % global_param)

# print("local_parm does not exit" % local_parm)
print("value of global_parm is %d" % global_param)



# 文件操作
'''
   r 以只读方式打开文件，该文件必须存在。

　　r+ 以可读写方式打开文件，该文件必须存在。

　　rb+ 读写打开一个二进制文件，只允许读写数据。

　　rt+ 读写打开一个文本文件，允许读和写。

　　w 打开只写文件，若文件存在则文件长度清为0，即该文件内容会消失。若文件不存在则建立该文件。

　　w+ 打开可读写文件，若文件存在则文件长度清为零，即该文件内容会消失。若文件不存在则建立该文件。

　　a 以附加的方式打开只写文件。若文件不存在，则会建立该文件，如果文件存在，写入的数据会被加到文件尾，即文件原先的内容会被保留。（EOF符保留）

　　a+ 以附加方式打开可读写的文件。若文件不存在，则会建立该文件，如果文件存在，写入的数据会被加到文件尾后，即文件原先的内容会被保留。 （原来的EOF符不保留）

　　wb 只写打开或新建一个二进制文件；只允许写数据。

　　wb+ 读写打开或建立一个二进制文件，允许读和写。

　　wt+ 读写打开或着建立一个文本文件；允许读写。

　　at+ 读写打开一个文本文件，允许读或在文本末追加数据。

　　ab+ 读写打开一个二进制文件，允许读或在文件末追加数据。



    转义字符	输出
        \'　'
        \"  "
        \a　‘bi’响一声
        \b	退格
        \f　	换页（在打印时）
        \n	回车，光标在下一行
        \r	换行，光标在上一行
        \t	八个空格(对齐)
        \\	\
        
'''
game_data = {"position": "N2 E3", "pocket": ["key", "knife"], "money": 160}
save_file = open("save.dat", "wb")  ## 其中形式有'w':write;'r':read.
pickle.dump(game_data, save_file)
save_file.close()  ## 关闭文件

load_file = open("save.dat", "rb")
load_game_data = pickle.load(load_file)
load_file.read()
load_file.readline()  # 读取第一行
load_file.readlines()  # python_list 形式
print(load_game_data)
load_file.close()

for x, y in [(1, 1), (2, 4), (3, 9)]:
    print(x, y)

# for循环后面还可以加上if判断，这样我们就可以筛选出仅偶数的平方：
[x * x for x in range(1, 11) if x % 2 == 0]

[m + n for m in 'ABC' for n in 'XYZ']

import os  # 导入os模块，模块的概念后面讲到
[d for d in os.listdir('.')]  # os.listdir可以列出文件和目录


## 面对对象编程
class Calculator:       #首字母要大写，冒号不能缺
    name = 'Good Calculator'  #该行为class的属性
    price = 18
    def add(self,x,y):
        print(self.name)
        result = x + y
        print(result)
    def minus(self,x,y):
        result=x-y
        print(result)
    def times(self,x,y):
        print(x*y)
    def divide(self,x,y):
        print(x/y)

'''
    cal=Calculator()  #注意这里运行class的时候要加"()",否则调用下面函数的时候会出现错误,导致无法调用.
    cal.name
    cal.add(10,20)
'''


# __init__可以理解成初始化class的变量，取自英文中initial 最初的意思.可以在运行时，给初始值附值，
#   这里的下划线是双下划线
#
#
# 运行c=Calculator('bad calculator',18,17,16,15),然后调出每个初始值的值。看如下代码。

class Calculator:
    name = 'good calculator'
    price = 18
    def __init__(self,name,price,height,width,weight):   # 注意，这里的下划线是双下划线
        self.name=name
        self.price=price
        self.h=height
        self.wi=width
        self.we=weight

## 异常处理
try:
    file=open('eeee.txt','r')  #会报错的代码
except Exception as e:  # 将报错存储在 e 中
    print(e)

## ZIP函数
a=[1,2,3]
b=[4,5,6]
ab=zip(a,b)
print(list(ab)) ## 需要list 来可视化这个功能
for i,j in zip(a,b):
     print(i/2,j*2)

## map
def fun(x,y):
	return (x+y)
list(map(fun,[1],[2]))

