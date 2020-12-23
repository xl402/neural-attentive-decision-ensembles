import numpy as np
import tensorflow as tf

from nade.networks.model import NADE


def test_nade_can_predict():
    model = NADE(units=2)
    x = tf.random.uniform((5, 100), dtype='float32')
    y = model(x)
    assert all(np.array(y.shape)==(5, 2))
