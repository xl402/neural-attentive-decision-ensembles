import tensorflow as tf

from node.networks.layer import ObliviousDecisionTree as ODT
from nade.networks.layer import AttentionMemory


class NADE(tf.keras.Model):
    def __init__(self,
                 units=1,
                 n_trees=3,
                 tree_depth=4,
                 num_heads=3,
                 memory_size=2,
                 link=tf.identity,
                 threshold_init_beta=1):

        super(NADE, self).__init__()
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.units = units,
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.link = link
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
        return self.link(h)


if __name__=='__main__':
    nade = NADE(units=1,
                n_trees=10,
                tree_depth=4,
                num_heads=3,
                memory_size=2)
    x = tf.random.uniform((1, 2))
    out = nade(x)
