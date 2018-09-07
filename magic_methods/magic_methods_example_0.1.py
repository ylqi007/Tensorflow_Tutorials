"""
`__getattr__`:

    This method will allow you to “catch” references to attributes that don’t exist in your object.
    But if the attribute does exist, `__getattr__` won't be invoked.
"""


class Dummy(object):
    pass


d = Dummy()
# d.does_not_exist  # Fails with AttributeError


class Dummy1(object):
    def __getattr__(self, attr):
        return attr.upper()


d1 = Dummy1()
print(d1.does_not_exist)        # 'DOES_NOT_EXIST'
print(d1.what_about_this_one)   # 'WHAT_ABOUT_THIS_ONE'
d1.value = "Python"
print(d1.value)