import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

from node.networks.layer import ObliviousDecisionTree as ODT


class AttentionMemory(tf.keras.layers.Layer):
    def __init__(self,
                 memory_size=2,
                 num_heads=3,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.attention = MultiHeadAttention(self.num_heads, key_dim=2)

    def build(self, input_shape):
        memory_initializer = tf.keras.initializers.random_uniform()
        memory_shape = (self.memory_size, input_shape[-1])
        init_value = memory_initializer(memory_shape, dtype='float32')
        self.memory = tf.Variable(init_value, trainable=True)

    def _broadcast_memory_along_batch(self, inputs):
        ones = tf.ones_like(inputs)
        ones = tf.expand_dims(inputs, axis=1)
        ones = tf.repeat(ones, self.memory_size, axis=1)
        memory = ones * self.memory
        return memory

    def call(self, inputs, training=None):
        x = tf.expand_dims(inputs, axis=1)
        memory = self._broadcast_memory_along_batch(inputs)
        m = tf.concat([x, memory], axis=1)
        m = self.attention(m, m)
        self.memory.assign(m[-1, 1:, :])
        x = m[:, 0, :]
        return x
