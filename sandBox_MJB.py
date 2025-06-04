squares = [1 , 2, 5, 77]
print(squares)
foo = squares
print(f"foo {foo}")
#squares = 44
squares[3] = 44
#squares[3] = 4
print(f"foo: {type(foo)} {foo}")

def foo(bar:int):
    print(bar)


foo(5)