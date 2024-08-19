#!/usr/bin/env python
# coding: utf-8

# # QUESTION-1

# Write a program(WAP) using loops and recursion:

# a) Factorial of n where n is a non negative integers

# In[14]:


#Factorial of n where n is a non negative integers using recursion
def fact_r(n):
    if n < 1:
        print("given number is negative")
        return 0
    if n == 1:
        return 1
    else:
        return n*fact_r(n-1)
n = int(input("enter a value for finding factorial : "))
print("factorial of",n,"is",fact_r(n))


# In[15]:


#Factorial of n where n is a non negative integers using iteration
def fact_i(n):
    num = 1
    for i in range(1,n+1):
        num *= i
    return num
    
n = int(input("enter a value for finding factorial : "))
print("factorial of",n,"is",fact_i(n))


# b) For calculating the Nth Fibonacci number.

# In[1]:


#For calculating the Nth Fibonacci number using recursion
def fib_r(n):
    if n <= 1:
        return n
    else:
        return fib_r(n - 1) + fib_r(n - 2)
n = int(input("enter a value for finding factorial : "))
print(n,"nth fibonacci number is",fib_r(n))


# In[2]:


#For calculating the Nth Fibonacci number using iteration
def fib_i(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
n = int(input("enter a value for finding factorial : "))
print("factorial of",n,"is",fib_i(n))


# c) To calculate a^b where a>0, b>=0

# In[6]:


def pow_r(a, b):
    if b == 0:
        return 1
    elif b == 1:
        return a
    elif b % 2 == 0:
        return pow_r(a * a, b // 2)
    else:
        return a * pow_r(a * a, b // 2)

a = int(input("enter a"))
b = int(input("enter b"))
print("Result is :",pow_r(a,b))


# In[7]:


def pow_i(a, b):
    result = 1
    while b > 0:
        if b % 2 == 1:
            result *= a
        a *= a
        b //= 2
    return result

a = int(input("enter a"))
b = int(input("enter b"))
print("Result is :",pow_i(a,b))


# 
# # QUESTION-2

# Query for 2 integers N and M from the user where 0<=N<=100 and 0<=M<=9. These will be the inputs to your function. Using recursion, compute the number of times the integer M occurs in all non-negative integers less than or equal to N.
# example: For N=13 and M=1, count=6 (numbers 1,10,11,12,13).

# In[11]:


def count_occurrences(N, M):
    count = 0
    for i in range(N + 1):
        count += str(i).count(str(M))
    return count

# Getting input from the user
while True:
    try:
        N = int(input("Enter an integer N (0 <= N <= 100): "))
        if 0 <= N <= 100:
            break
        else:
            print("N must be between 0 and 100.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

while True:
    try:
        M = int(input("Enter an integer M (0 <= M <= 9): "))
        if 0 <= M <= 9:
            break
        else:
            print("M must be between 0 and 9.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

# Computing the count of occurrences
occurrences = count_occurrences(N, M)
print("Number of times", M, "occurs in all non-negative integers less than or equal to", N, "is:", occurrences)


# # QUESTION-3

# Programs using lambda function.

# a) Given a list of names, use map to create a list where each name is prefixed with "Hello, ".
# 
# Example Input: ['Alice', 'Bob', 'Charlie']
# 
# Example Output: ['Hello, Alice', 'Hello, Bob', 'Hello, Charlie']

# In[12]:


lst1 = ["venky","vinod","pritam"]
# Using map with a lambda function to prefix each name with "Hello, "
greetings = list(map(lambda lst1: "Hello, " + lst1, lst1))

print(greetings)


# b) Use filter and a lambda function to extract all even numbers from a given list.
# 
# Example Input: [1, 2, 3, 4, 5, 6]
# 
# Example Output: [2, 4, 6]

# In[17]:


lst2 = [1, 2, 3, 4, 5, 6]
even_numbers = list(filter(lambda x: x % 2 == 0,lst2))
print(even_numbers)


# c) Use reduce and lambda to concatenate all strings in a given list.
# 
# Example Input: ['Python', 'is', 'awesome']
# 
# Example Output: 'Pythonisawesome'

# In[18]:


from functools import reduce

strings = ['Python', 'is', 'awesome']

# Using reduce with a lambda function to concatenate all strings
concatenated_string = reduce(lambda x, y: x + y, strings)

print(concatenated_string)


# # QUESTION-4

# Define a class Complex that defines a complex number with attributes real and imaginary (as we did in the class). Define operators for addition, subtraction, multiplication and division (Do with both operator overloading as well as without overloading). While printing the output, print in the form of complex number form like ( a + ib)  - 10 marks ( 1 mark each for each of the operations with and without operator overloading)

# In[19]:


class Complex:
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary

# Addition with operator overloading
    def __add__(self, other):
        real_part = self.real + other.real
        imaginary_part = self.imaginary + other.imaginary
        return Complex(real_part, imaginary_part)

# Subtraction with operator overloading
    def __sub__(self, other):
        real_part = self.real - other.real
        imaginary_part = self.imaginary - other.imaginary
        return Complex(real_part, imaginary_part)

# Multiplication with operator overloading
    def __mul__(self, other):
        real_part = self.real * other.real - self.imaginary * other.imaginary
        imaginary_part = self.real * other.imaginary + self.imaginary * other.real
        return Complex(real_part, imaginary_part)

# Division with operator overloading
    def __truediv__(self, other):
        denominator = other.real ** 2 + other.imaginary ** 2
        real_part = (self.real * other.real + self.imaginary * other.imaginary) / denominator
        imaginary_part = (self.imaginary * other.real - self.real * other.imaginary) / denominator
        return Complex(real_part, imaginary_part)

# Addition without operator overloading
    def add(self, other):
        real_part = self.real + other.real
        imaginary_part = self.imaginary + other.imaginary
        return Complex(real_part, imaginary_part)

# Subtraction without operator overloading
    def subtract(self, other):
        real_part = self.real - other.real
        imaginary_part = self.imaginary - other.imaginary
        return Complex(real_part, imaginary_part)

# Multiplication without operator overloading
    def multiply(self, other):
        real_part = self.real * other.real - self.imaginary * other.imaginary
        imaginary_part = self.real * other.imaginary + self.imaginary * other.real
        return Complex(real_part, imaginary_part)

# Division without operator overloading
    def divide(self, other):
        denominator = other.real ** 2 + other.imaginary ** 2
        real_part = (self.real * other.real + self.imaginary * other.imaginary) / denominator
        imaginary_part = (self.imaginary * other.real - self.real * other.imaginary) / denominator
        return Complex(real_part, imaginary_part)

    def __str__(self):
        return f"({self.real} + {self.imaginary}i)"


# Test
c1 = Complex(2, 3)
c2 = Complex(1, 2)

# With operator overloading
print("With operator overloading:")
print("Addition:", c1 + c2)
print("Subtraction:", c1 - c2)
print("Multiplication:", c1 * c2)
print("Division:", c1 / c2)

# Without operator overloading
print("\nWithout operator overloading:")
print("Addition:", c1.add(c2))
print("Subtraction:", c1.subtract(c2))
print("Multiplication:", c1.multiply(c2))
print("Division:", c1.divide(c2))


# In[ ]:




