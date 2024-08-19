#!/usr/bin/env python
# coding: utf-8

# # Question-5

# WAP to implement a class called "Bank_Account" representing a person's bank account.
# 
# The class should have the following attributes: account_holder_name (str), account_number(int), address (str) and balance (float).
# 
# The class should have methods to implement the following:
# 
#     deposit - Deposits a given amount into the account
#     
#     withdraw - Withdraws a given amount from the account
#     
#     check_balance - To get the current balance
#     
#     update_details - To update the name and address from the user and displays a message indicating successful update
#     display_details - To display the details of the account.

# In[101]:


class Bank_Account:
    def __init__(self,name,number,address,balance):
        self.name = name
        self.number = number
        self.address = address
        self.balance = balance
        
    def Deposit(self,amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposit {amount} into {self.number}, updated balance is {self.balance}")
        else:
            print("Enter a valid amount to deposit")
                
    def Withdraw(self,amount):
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            print(f"Withdraw {amount} from {self.number}, updated balance is {self.balance}")
        else:
            print("Enter a valid amount to withdraw")
                
    def Check_balance(self):
        print(f"Current balance is {self.balance}")
        
    def Update_details(self,new_name,new_address):
        self.name = new_name
        self.address = new_address
        print("Details updated successfully")
        
    def Display(self):
        print(f"Account holder name is :{self.name}")
        print(f"Account number is :{self.number}")
        print(f"Address of the Account holder is :{self.address}")
        print(f"Balance in the Account is: {self.balance}")

# Example usage:
account1 = Bank_Account("Venky", 34532558874, "3-31,mugada",40000)
print("Initial Account Details:")
account1.Display()


# In[102]:


account1.Deposit(50000)


# In[103]:


account1.Withdraw(25000)


# In[104]:


account1.Update_details("sai","3-23,mugada")


# In[105]:


account1.Display()


# # Question-6

# Define a Matrix class of dimensions m X n (the values for m and n can be taken as input). Demonstrate matrix addition, subtraction, multiplication, element-by-element multiplication, scalar multiplication (use map here). Use operator overloading wherever possible. (Hint: In the constructor, use 'random' and create list of list using list comprehension. In the arguments of constructor, send the number of rows and columns)
# Ensure that your implementation follows best practices for class design and encapsulation in Python. Comment your code to explain the functionality of each method.

# In[106]:


import random

class Matrix:
    def __init__(self, m, n):
        self.rows = m
        self.cols = n
        self.data = [[random.randint(0, 9) for _ in range(n)] for _ in range(m)]
    
    def __repr__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            print("Matrices must have the same dimensions for addition")
            return None
        
        result = Matrix(self.rows, self.cols)
        result.data = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return result
    
    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            print("Matrices must have the same dimensions for subtraction")
            return None
        
        result = Matrix(self.rows, self.cols)
        result.data = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return result
    
    def __mul__(self, other):
        if self.cols != other.rows:
            print("Number of columns in the first matrix must be equal to the number of rows in the second matrix")
            return None
        
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                result.data[i][j] = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
        return result
    
    def elementwise_multiply(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            print("Matrices must have the same dimensions for element-wise multiplication")
            return None
        
        result = Matrix(self.rows, self.cols)
        result.data = [[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return result
    
    def scalar_multiply(self, scalar):
        result = Matrix(self.rows, self.cols)
        result.data = [[scalar * self.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return result

# Example usage
m = int(input("Enter the number of rows: "))
n = int(input("Enter the number of columns: "))

matrix1 = Matrix(m, n)
matrix2 = Matrix(m, n)

print("Matrix 1:")
print(matrix1)
print("Matrix 2:")
print(matrix2)


# In[107]:


print("Matrix Addition:")
addition_result = matrix1 + matrix2
if addition_result:
    print(addition_result)


# In[108]:


print("Matrix Subtraction:")
subtraction_result = matrix1 - matrix2
if subtraction_result:
    print(subtraction_result)


# In[109]:


print("Matrix Multiplication:")
multiplication_result = matrix1 * matrix2
if multiplication_result:
    print(multiplication_result)


# In[110]:


print("Element-wise Multiplication:")
elementwise_result = matrix1.elementwise_multiply(matrix2)
if elementwise_result:
    print(elementwise_result)


# In[111]:


scalar = int(input("Enter the scalar value for multiplication: "))
print("Scalar Multiplication:")
print(matrix1.scalar_multiply(scalar))


# # Question-7 

# Create a Python class named Time that represents a moment of time. The class should have attributes hour, minute, and second. Include the following features:
# 
#     A constructor that initializes hour, minute, and second, with validation to ensure each attribute is within its correct range (hours: 0-23, minutes: 0-59, seconds: 0-59).
#     
#     A __str__() method that returns the time in a format hh:mm:ss.
#     
#     Methods set_time(hour, minute, second) and get_time() to update and access the time, respectively.
#     
#     An add_seconds(seconds) method that adds a given number of seconds to the current time object, adjusting the hour, minute, and second attributes accordingly.

# In[112]:


class Time:
    def __init__(self, hour=0, minute=0, second=0):
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.set_time(hour, minute, second)

    def set_time(self, hour, minute, second):
        if 0 <= hour < 24:
            self.hour = hour
        else:
            print("Hour must be in the range 0-23")
        if 0 <= minute < 60:
            self.minute = minute
        else:
            print("Minute must be in the range 0-59")
        if 0 <= second < 60:
            self.second = second
        else:
            print("Second must be in the range 0-59")

    def get_time(self):
        return self.hour, self.minute, self.second

    def __str__(self):
        return f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}"

    def add_seconds(self, seconds):
        total_seconds = self.second + seconds
        self.second = total_seconds % 60
        total_minutes = self.minute + total_seconds // 60
        self.minute = total_minutes % 60
        self.hour = (self.hour + total_minutes // 60) % 24

# Example usage
time = Time(10, 30, 45)
print("Current Time:", time)


# In[113]:


time.add_seconds(75)
print("Time after adding 75 seconds:", time)


# In[114]:


time.set_time(23, 59, 59)
print("Current Time:", time)


# In[115]:


time.add_seconds(2)
print("Time after adding 2 seconds:", time)


# # Question-8

# Create a class named Geometry that provides static methods for various geometric calculations, such as area and perimeter, for different shapes (circle, rectangle, square). Include:
# 
# Static methods like circle_area(radius), rectangle_area(length, width), and square_area(side).
# 
# Static methods for perimeter calculations for the above shapes.
# 
# Ensure that methods check for valid inputs (e.g., positive numbers).

# In[117]:


import math

class Geometry:
    def circle_area(radius):
        area = None
        if radius > 0:
            area = math.pi * radius ** 2
        else:
            print("Radius must be a positive number")
        return area

    def rectangle_area(length, width):
        area = None
        if length > 0 and width > 0:
            area = length * width
        else:
            print("Length and width must be positive numbers")
        return area
    
    def square_area(side):
        area = None
        if side > 0:
            area = side ** 2
        else:
            print("Side length must be a positive number")
        return area

    def circle_perimeter(radius):
        perimeter = None
        if radius > 0:
            perimeter = 2 * math.pi * radius
        else:
            print("Radius must be a positive number")
        return perimeter

    def rectangle_perimeter(length, width):
        perimeter = None
        if length > 0 and width > 0:
            perimeter = 2 * (length + width)
        else:
            print("Length and width must be positive numbers")
        return perimeter

    def square_perimeter(side):
        perimeter = None
        if side > 0:
            perimeter = 4 * side
        else:
            print("Side length must be a positive number")
        return perimeter


radius = 5
circle_area = Geometry.circle_area(radius)
if circle_area is not None:
    print("Circle Area:", circle_area)


# In[118]:


length, width = 4, 6
rectangle_area = Geometry.rectangle_area(length, width)
if rectangle_area is not None:
    print("Rectangle Area:", rectangle_area)


# In[119]:


side = 3
square_area = Geometry.square_area(side)
if square_area is not None:
    print("Square Area:", square_area)


# In[120]:


circle_perimeter = Geometry.circle_perimeter(radius)
if circle_perimeter is not None:
    print("Circle Perimeter:", circle_perimeter)


# In[121]:


rectangle_perimeter = Geometry.rectangle_perimeter(length, width)
if rectangle_perimeter is not None:
    print("Rectangle Perimeter:", rectangle_perimeter)


# In[122]:


square_perimeter = Geometry.square_perimeter(side)
if square_perimeter is not None:
    print("Square Perimeter:", square_perimeter)


# In[ ]:




