#!/usr/bin/env python
# coding: utf-8

# # QUESTION-6

# Create a base class Vehicle with the following attributes:
# 
#          make (string)
# 
#          model (string)
# 
#          year (int)
# 
#        Create a method initialize_vehicle to set the above attributes. Also, create a method display_vehicle to print these attributes.
# 
#        Create a class Car inherited from Vehicle with the following additional attribute:
# 
#          fuel_type (string)
# 
#        Create a method get_car_details to initialize the above attribute along with Vehicle attributes.
# 
#        Also, create a method display_vehicle to print these attributes along with Vehicle attributes.
# 
#        Create a class Bike inherited from Vehicle with the following additional attribute:
# 
#            gear_count (int)
# 
#       Create a method get_bike_details to initialize the above attribute along with Vehicle attributes.
# 
#       Also, create a method display_vehicle to print these attributes along with Vehicle attributes.
# 
#       Create two different objects for Car and Bike and demonstrate each of the methods.
# 
#      Example -1:
# 
#          my_car = Car()
# 
#         my_car.get_car_details("Toyota", "Camry", 2020, "Petrol")
# 
#         my_car.display_vehicle()
# 
#         Output:
# 
#            Make: Toyota, Model: Camry, Year: 2020
# 
#            Fuel Type: Petrol
# 
#           Example -2 :
# 
#            my_bike = Bike()
# 
#            my_bike.get_bike_details("Yamaha", "YZF R1", 2021, 6)
# 
#            my_bike.display_vehicle()
# 
#           Output:
# 
#              Make: Yamaha, Model: YZF R1, Year: 2021
# 
#              Gear Count: 6
# 
# 

# In[12]:


class Vehicle:
    def initialize_vehicle(self,make,model,year):
        self.make = make
        self.model = model
        self.year = year
        
    def display_vehicle(self):
        print(f"make :{self.make},\n model is :{self.model},\n year is :{self.year}")
        
class Car(Vehicle):
    def get_car_details(self, make, model, year, fuel_type):
        self.initialize_vehicle(make, model, year)
        self.fuel_type = fuel_type

    def display_vehicle(self):
        super().display_vehicle()
        print(f"Fuel Type: {self.fuel_type}")
        
my_car = Car()
my_car.get_car_details("Toyota", "Camry", 2020, "Petrol")
my_car.display_vehicle()


# In[13]:


#Example 2
class Bike(Vehicle):
    def get_bike_details(self, make, model, year, gear_count):
        self.initialize_vehicle(make, model, year)
        self.gear_count = gear_count

    def display_vehicle(self):
        super().display_vehicle()
        print(f"Gear Count: {self.gear_count}")
        
my_bike = Bike()
my_bike.get_bike_details("Yamaha", "YZF R1", 2021, 6)
my_bike.display_vehicle()


# # QUESTION-7

# Suppose you are building a Python program to manage a school's student data. You need to create a Student class that contains information such as the student's name, age, grade, and class schedule. Additionally, there are some attributes that are shared by all students, such as the school name, the total number of students, and the number of classes offered.
# 
# How can you use class variables in Python to define these shared attributes of the Student class? What are the advantages of using class variables in this scenario? Can you provide an example program that demonstrates the use of class variables in the Student class? 

# In[11]:


class Student:
    school_name = "IITM"
    total_students = 0
    total_classes = 5  

    def __init__(self, name, age, grade, schedule):
        self.name = name
        self.age = age
        self.grade = grade
        self.schedule = schedule
        Student.total_students += 1 #incriminant operation for adding new student in the class

    def display_student_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Grade: {self.grade}")
        print(f"Class Schedule: {self.schedule}")
        print(f"School Name: {self.school_name}")
        print(f"Total Students: {self.total_students}")
        print(f"Total Classes Offered: {self.total_classes}")


student1 = Student("Venky", 23, 10, ["Math", "Science", "Computer"])
student2 = Student("Vinod", 25, 11, ["English", "Mathematics", "Chemistry"])

student1.display_student_info()
student2.display_student_info()


# # QUESTION-8

# Class Inheritance in Python: Finding GCD (greatest common divisor) and LCM (least common multiple) of Numbers and Handling Composite Numbers.
# 
# 
# a) Create a Numbers class with a, b, find_gcd(), and find_lcm() methods.
# 
# b) Create an EvenNumbers class that inherits from Numbers and overrides find_lcm() to handle even numbers.
# 
# c) Create an OddNumbers class that inherits from Numbers and overrides find_lcm() to handle odd numbers.
# 
# d) Create a CompositeNumbers class that inherits from EvenNumbers and OddNumbers and overrides find_gcd() to handle composite numbers.
# 
# e) Create a CompositeNumbers object with a = 12 and b = 9, and call its find_lcm() and find_gcd() methods.

# In[8]:


class Numbers:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def gcd(self):
        x, y = self.a, self.b
        while y:
            x, y = y, x % y
        return x

    def lcm(self):
        return (self.a * self.b) // self.gcd()


class EvenNumbers(Numbers):
    def find_lcm(self):
        if self.a % 2 == 0 and self.b % 2 == 0:
            return super().lcm()
        else:
            print("LCM is not defined for odd numbers.")


class OddNumbers(Numbers):
    def find_lcm(self):
        if self.a % 2 != 0 and self.b % 2 != 0:
            return super().lcm()
        else:
            print("LCM is not defined for even numbers.")


class CompositeNumbers(EvenNumbers, OddNumbers):
    def find_gcd(self):
        return super().gcd()

composite_obj = CompositeNumbers(12,9)
print("GCD of given numbers is :", composite_obj.gcd())
print("LCM of given numbers is :",composite_obj.lcm())


# # QUESTION-9

# WAP to manage the collections of books in a library in the following  manner:
# 
#  Create a Python script that can both read from and write to a CSV file, containing details about each book. Each book's information will include its title, author, publication year, and ISBN number. Your script should be capable of adding new books to the CSV file and listing all the books currently stored in the file.
# 
# The program should begin by checking if the CSV file exists. If it does not, your script should create it and initialize it with the appropriate headers. Then, there should be 2 options: to add a new book or to display all books. When adding a new book, the user should be prompted to enter the title, author, publication year, and ISBN number. This new book should then be added to the CSV file without overwriting the existing entries. When choosing to display all books, the script should read from the CSV file and print each book's details.

# In[1]:


import csv
import os


def create_csv_file(file_name): #Creating a new file if it won't exist
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Author", "Publication Year", "ISBN"])

def add_book(file_name, book_details): # To add a new book to the CSV file
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(book_details)

def display_books(file_name): # To display all books stored in the CSV file
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)

file_name = "books.csv"

if not os.path.exists(file_name):
    create_csv_file(file_name)

while True:
    print("\n1. Add a new book")
    print("2. Display all books")
    print("3. Quit")
    choice = input("Enter your choice: ")

    if choice == '1':
        title = input("Enter the title of the book: ")
        author = input("Enter the author of the book: ")
        publication_year = input("Enter the publication year of the book: ")
        isbn = input("Enter the ISBN of the book: ")
        book_details = [title, author, publication_year, isbn]
        add_book(file_name, book_details)
        print("Book added successfully!")
    elif choice == '2':
        print("\nList of all books:")
        display_books(file_name)
    elif choice == '3':
        print("Exiting program.")
        break
    else:
        print("Invalid choice. Please enter a valid option.")


# # QUESTION-10

# WAP to create a pandas dataframe with a list of words and sort them in ascending order. The sorted words should be copied to a new file.

# In[9]:


import pandas as pd

words = ['VENKY','VINOD','SOURAV','PRABHATH']

df = pd.DataFrame(words, columns=['Words'])

df_sorted = df.sort_values(by='Words')

output_file = 'sorted_words.txt'

df_sorted.to_csv(output_file, index=False, header=False)

print("Sorted words have been written to", output_file)


# In[14]:


with open('sorted_words.txt', 'r') as file:
    for line in file:
        print(line.strip())


# In[ ]:




