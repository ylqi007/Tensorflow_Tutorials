"""
Make ecah element in the tuple is accessed as an attribute, with the first element
being the attribute `_1`, the second `_2`, and so on.
"""


class Tuple(tuple):
    def __getattr__(self, item):
        def _int(val):
            try:
                return int(val)
            except ValueError:
                return False

        if not item.startswith('_') or not _int(item[1:]):
            raise AttributeError("`tuple` object has no attribute '%s'" % item)
        index = _int(item[1:]) - 1
        return self[index]


t = Tuple(['z', 3, 'Python', -1])
print(t._1)     # item='_1', item[1:]='1'
print(t._2)
print(t._3)
print(t._4)

