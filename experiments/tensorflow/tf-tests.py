# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:30:53 2021

@author: caleu
"""

import tensorflow as tf

print('\n------Line clear-----\n')

# print("this is the current tensorflow version: ")
# print(str(tf.version) + '\n')

rank1_tensor = tf.Variable(["yoyoy", "ayaya", "test", "alright"], tf.string)
rank2_tensor = tf.Variable([["yoyoy", "ayaya"], ["test", "alright"]], tf.string)

print(str(tf.rank(rank1_tensor)))
print(str(tf.rank(rank2_tensor)))

print("tensor rank 2 shape")
print(rank2_tensor.shape)

tensor1 = tf.ones([1, 2, 3]) # tf.ones() creates a tensor with the given shape, populated with ones

tensor2 = tf.reshape(tensor1, [2,3,1]) # reshapes an existing tensor to the given shape
tensor3 = tf.reshape(tensor2, [2, -1]) # the -1 tells the function to calculate the dimension in its place
                                       # In this case, it should be a 3, making the shape [2, 3]

print(tensor1)
print(tensor2)
print(tensor3)