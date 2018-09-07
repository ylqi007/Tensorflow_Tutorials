"""
`__getattribute__` is similar to `__getattr__`, with the important difference
that `__getattribute__` will intercept EVERY attribute lookup, doesn’t matter if the attribute exists or not.
It means that we’ve virtually lost the value attribute; it has become “unreachable”.
"""


class Dummy(object):
    def __getattribute__(self, attr):
        return 'YOU SEE ME?'


d = Dummy()
d.value = "Python"
d.value1 = "Java"
print(d.value)  # "YOU SEE ME?"
print(d.value1)


print("##############################")


class Dummy1(object):
    def __getattribute__(self, attr):
        __dict__ = super(Dummy1, self).__getattribute__('__dict__')
        if attr in __dict__:
            return super(Dummy1, self).__getattribute__(attr)
        return attr.upper()


d1 = Dummy1()
d1.value = "Python"
print(d1.value)  # "Python"
print(d1.does_not_exist)  # "DOES_NOT_EXIST"