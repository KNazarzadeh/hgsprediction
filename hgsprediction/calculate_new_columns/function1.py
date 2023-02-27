import numpy as np

class EXAMPLE1():

    def __init__(
        self,
        x: int = 1,
        y: int = 2,
        method: str = "mean"
    ):

        self.x = x
        self.y = y
        self.method = method

    def compute(self, x, y):

        method = self.method

        assert isinstance(x, int), "x must be integer!"
        assert isinstance(y, int), "y must be integer!"
        assert isinstance(method, str), "method must be integer!"

        if(method == "mean"):
            z = (x + y)/2
        else:
            z = (x-y)/2

        return z