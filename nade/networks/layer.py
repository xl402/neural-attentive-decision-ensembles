import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention

from node.networks.layer import ObliviousDecisionTree as ODT


class AttentiveDecisionTree(tf.keras.layers.Layer):
    def __init__(self,
                 units=1,
                 n_trees=3,
                 tree_depth=4,
                 num_heads=3,
                 memory_size=5,
                 threshold_init_beta=1):

        super(AttentiveDecisionTree, self).__init__()
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.units = units,
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.threshold_init_beta = threshold_init_beta

        self.memory_block = AttentionMemory(memory_size, num_heads)
        self.tree = ODT(n_trees=n_trees,
                        depth=tree_depth,
                        units=units,
                        threshold_init_beta=threshold_init_beta)

    def call(self, inputs, training=None):
        x_hat = self.memory_block(inputs)
        x = inputs + x_hat
        h = self.tree(x)
        return h


class AttentionMemory(tf.keras.layers.Layer):
    def __init__(self,
                 memory_size=2,
                 num_heads=3,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.attention = MultiHeadAttention(num_heads,
                                            key_dim=2,
                                            attention_axes=1)

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
        # TODO: need to use positional encoding for next step
        self.memory.assign(m[-1, 1:, :])
        x = m[:, 0, :]
        return x


if __name__=='__main__':
    x = tf.random.uniform((100000, 2))
    y = x + 2
    layer = AttentionMemory()
    inputs = tf.keras.Input(2)
    outputs =layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    outputs = layer(inputs)

    model.compile(optimizer='adam',
                  loss='mse')
    model.fit(x, y, batch_size=1)
