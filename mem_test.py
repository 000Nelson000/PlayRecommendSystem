# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:22:47 2017

@author: 116952
"""
#%%
#import memory_profiler
from memory_profiler import profile

@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()