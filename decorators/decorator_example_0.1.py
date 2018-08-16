"""
So, @my_decorator is just an easier way of saying

    `just_some_function=my_decorator(just_some_function)`

It's how you apply a decorator to a function.
"""
# 2018-08-15 -- Qi


def my_decorator(some_function):

    def wrapper():
        num = 10
        if num == 10:
            print("Yes!")
        else:
            print("No!")
        some_function()
        print("Something is happening after some_function() is called.")
    return wrapper


@my_decorator
def just_some_function():
    print("Wheee!")


just_some_function()