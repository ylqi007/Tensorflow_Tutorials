* `__getattr__`:

This method will allow you to “catch” references to attributes that don’t exist in your object.
But if the attribute does exist, `__getattr__` won't be invoked.


* `__getattribute__`

`__getattribute__` is similar to `__getattr__`, with the important difference
that `__getattribute__` will intercept EVERY attribute lookup, doesn’t matter if the attribute exists or not.
It means that we’ve virtually lost the value attribute; it has become “unreachable”.
