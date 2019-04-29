# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:52:29 2019

@author: Admin
"""

from tkinter  import  *

def show(event):
      s = '滑块的取值为' + str(var.get())
      print(var.get())
      lb.config(text=s)

root = Tk()
root.title('滑块实验')
root.geometry('320x180')
var=DoubleVar()
print(var)
scl1 = Scale(root,orient=HORIZONTAL,length=200,from_=1.0,to=5.0,label='请拖动滑块',tickinterval=1,resolution=0.05,variable=var)
scl1.bind('<ButtonRelease-1>',show)#绑定鼠标释放 显示函数
scl1.pack()

scl2 = Scale(root,orient=HORIZONTAL,length=200,from_=1.0,to=5.0,label='请拖动滑块',tickinterval=1,resolution=0.05,variable=var)
scl2.bind('<ButtonRelease-1>',show)#绑定鼠标释放 显示函数
scl2.pack()

scl3 = Scale(root,orient=HORIZONTAL,length=200,from_=1.0,to=5.0,label='请拖动滑块',tickinterval=1,resolution=0.05,variable=var)
scl3.bind('<ButtonRelease-1>',show)#绑定鼠标释放 显示函数
scl3.pack()

lb = Label(root,text='')
lb.pack()

root.mainloop()