import numpy as np
import tensorflow as tf

# a = np.array([[1,2], [3,4], [5,6]])
# b = np.array([[7,8], [9,10], [11,12]])
# c = np.array([[13,14], [15,16], [17,18]])
# d = np.array([[19,20], [21,22], [23,24]])
# print()
# b = np.array([7,8])
# c = np.array([3,4])

# a[0:2] = b

# print(a)

# a = tf.Tensor()
# x = []
# b = tf.constant(([[1,2,3]]))


# for i in range(3):
#     x.append(b[:,i])
# print(x)
# x = tf.stack(x)
# x= x
# print(x)

# logits = np.array([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]])
# print(logits.shape)
# print(tf.nn.softmax(logits)) 
# policy = [[0.8437947, 0.11419519, 0.04201007, 0.00657326, 0.9755587, 0.01786798]]
# labels = [[1.0, 0.0, 0.0, 0.0, 0.8, 0.2]]
# print(tf.multiply(tf.math.log(policy), labels))
# print(-tf.reduce_sum(tf.multiply(tf.math.log(policy), labels))) #
# logits = [4.0, 2.0, 1.0, 0.0, 5.0, 1.0]
# labels = [1.0, 0.0, 0.0, 0.0, 0.8, 0.2]
# # print(logits[0])
# # print(tf.nn.softmax(logits[0]))
# print(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)) #[0.16984604 0.82474494]
# # temp = 0
# # for i in range(2):
# #     temp += tf.nn.softmax_cross_entropy_with_logits(labels=labels[i], logits=logits[i])

# # print(temp)

# actions=[0,0,0,0,1,0,0,0,0,0]
# policy=[0.1, 0.1, 0.1, 0.1, 0.5, 0.0, 0.0,0.0, 0.0, 0.1]
# # print(tf.matmul(actions, policy))
# print(tf.maximum(policy, 0.15))

# actions = np.array()

a = np.array([[1]])
print(np.asscalar(a))