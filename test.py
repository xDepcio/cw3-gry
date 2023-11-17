class A:
    pass


class B(A):
    pass


print(isinstance(A(), A))  # True
print(isinstance(B(), A))  # True
