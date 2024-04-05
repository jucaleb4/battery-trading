import unittest
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from utils import SimpleLogger

class TestSimpleLogger(unittest.TestCase):
    def test_save_correctly(self):
        fname = os.path.join("logs", "test.csv")
        logger = SimpleLogger(fname, ["a", "b"])
        info = OrderedDict([("a", ...), ("b", ...), ("c", "red")])

        m = 100
        data = np.hstack((
            np.random.random((m,2)),
            np.atleast_2d(np.random.randint(0,3,size=m)).T,
            np.random.random((m,1))
        ))

        m = 100
        for d in data:
            info["a"] = d[0]
            info["b"] = d[1]
            logger.store((info, d[2], d[3]))

        logger.save()
        saved_data = pd.read_csv(fname, header="infer")

        # tolerance of 0.01 since we round off at 0.01
        self.assertTrue(np.allclose(data, saved_data, atol=1e-2))

        
