# AUTOGENERATED! DO NOT EDIT! File to edit: nbs\00_demo.ipynb (unless otherwise specified).

__all__ = ['test', 'test2', 'test3', 'SayTest']

# Cell
def test(a, b):
    return a+b

# Cell
def test2(a, b, c):
    return a+b+c

# Cell
def test3(test, test2):
    a = 1
    b = 2
    c = 3
    return test2(test(a,b), b, c)

# Cell
class SayTest(object):
    """ Some more remarks"""

    def __init__(self, multiplier):
        """set the multiplier"""
        self.multiplier = multiplier

    def __call__(self, number):
        """return number * multiplier"""
        return self.multiplier*number