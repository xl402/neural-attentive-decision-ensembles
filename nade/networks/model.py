import tensorflow as tf

from nade.networks.layer import AttentiveDecisionTree as ADT


class NADE(tf.keras.Model):
    def __init__(self,
                 units=1,
                 n_layers=3,
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
        self.n_layers = 3

        self.bn = tf.keras.layers.BatchNormalization()
        self.ensemble = [ADT(n_trees=n_trees,
                             num_heads=num_heads,
                             memory_size=memory_size,
                             tree_depth=tree_depth,
                             units=units,
                             threshold_init_beta=threshold_init_beta)
                         for _ in range(n_layers)]
        self.link = link

    def call(self, inputs, training=None):
        x = self.bn(inputs, training=training)
        h = 0.
        for tree in self.ensemble:
            h = h + tree(x)
            x = tf.concat([x, h], axis=1)
        return self.link(h)


if __name__=='__main__':
    nade = NADE(units=1,
                n_layers=3,
                n_trees=10,
                tree_depth=4,
                num_heads=3,
                memory_size=2)
    x = tf.keras.Input((10,))
    out = nade(x)
