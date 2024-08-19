#!/usr/bin/env python
# coding: utf-8

# # QUESTION-1

# Design a class named Polygon that initializes with the length of a side. Then, derive a class named Square from the Polygon class. Utilize the side length defined in the Polygon class for the Square class. Within the Square class, implement a method called findArea() that calculates and returns the square's area based on its side length. Use __init__() for necessary initialization

# In[122]:


class Polygon: 
    def __init__(self,length):#lenghth = length of a side in the polygon
        self.length = length
        
class Square(Polygon):
    def __init__(self,length):
        super().__init__(length)
        
    def Area(self):
        return self.length ** 2
    
square = Square(6) #we are calling "Area()" method on the variable Square.
print("Area of the square:", square.Area())


# # QUESTION-2

#  (a)Create a class Father with attributes
# 
#                     - father_name (string), father_age (int), father_talents (set of strings)
# 
#                 Create a class Mother with attributes:
# 
#                     - mother_name (string), mother_age (int), mother_talents (set of strings)
# 
#                  Create a class Child that inherits both father and mother with attributes: 
# 
#                     - child_name (string), child_age (int), child_gender(string)

# In[123]:


class Father:
    def __init__(self,father_name,father_age,father_talents):
        self.father_name = father_name
        self.father_age = father_age
        self.father_talents = father_talents
        
class Mother:
    def __init__(self,mother_name,mother_age,mother_talents):
        self.mother_name = mother_name
        self.mother_age = mother_age
        self.mother_talents = mother_talents
        
class Child(Father,Mother):
    def __init__(self,father_name,father_age,father_talents,mother_name,mother_age,mother_talents,child_name,child_age,child_gender):
        Father.__init__(self,father_name,father_age,father_talents)
        Mother.__init__(self,mother_name,mother_age,mother_talents)
        self.child_name = child_name
        self.child_age = child_age
        self.child_gender = child_gender
        
father = Father("Apparao",48,"Agriculture")
mother = Mother("Ravanamma",46,"Agriculture")
child = Child("Apparao",48,"Agriculture","Ravanamma",46,"Agriculture","venky",23,"Male")

print("Child Name:", child.child_name)
print("Child Age:", child.child_age)
print("Child Gender:", child.child_gender)
print("Father Name:", child.father_name)
print("Mother Name:", child.mother_name)


# (b) Create a function getChildDetails() in Child to input the details of the child, it’s father and mother and printChildDetails() to print the details using     printChildDetails()

# In[124]:


class Father:
    def __init__(self,father_name,father_age,father_talents):
        self.father_name = father_name
        self.father_age = father_age
        self.father_talents = father_talents
        
class Mother:
    def __init__(self,mother_name,mother_age,mother_talents):
        self.mother_name = mother_name
        self.mother_age = mother_age
        self.mother_talents = mother_talents
        
class Child(Father,Mother):
    def __init__(self,father_name,father_age,father_talents,mother_name,mother_age,mother_talents,child_name,child_age,child_gender):
        Father.__init__(self,father_name,father_age,father_talents)
        Mother.__init__(self,mother_name,mother_age,mother_talents)
        self.child_name = child_name
        self.child_age = child_age
        self.child_gender = child_gender
        
    def getChildDetails(self):
        self.child_name = input("Enter child name: ")
        self.child_age = int(input("Enter child age: "))
        self.child_gender = input("Enetr the child Gender: ")
        self.father_name = input("Enter Father name: ")
        self.father_age = int(input("Enter Father age: "))
        self.father_talents = input("Enter Father talents: ")
        self.mother_name = input("Enter Mother name: ")
        self.mother_age = int(input("Enter Mother age: "))
        self.mother_talents = input("Enter Mother talents: ")
        
    def printChildDetails(self):
        print("child name is : ",self.child_name)
        print("child age is : ",self.child_age)
        print("child gender is : ",self.child_gender)
        print("Father name is : ",self.father_name)
        print("Father age is : ",self.father_age)
        print("Father talents are : ",self.father_talents)
        print("Mother name is : ",self.mother_name)
        print("Mother age is : ",self.mother_age)
        print("Mother talents are : ",self.mother_talents)

child1 = Child("",0,set(),"",0,set(),"",0,"")
child1.getChildDetails()
child1.printChildDetails()


# (c) Create an object of class Child and read the details by invoking getChildDetails() and display the details entered.

# In[126]:


class Father:
    def __init__(self,father_name,father_age,father_talents):
        self.father_name = father_name
        self.father_age = father_age
        self.father_talents = father_talents
        
class Mother:
    def __init__(self,mother_name,mother_age,mother_talents):
        self.mother_name = mother_name
        self.mother_age = mother_age
        self.mother_talents = mother_talents
        
class Child(Father,Mother):
    def __init__(self,father_name,father_age,father_talents,mother_name,mother_age,mother_talents,child_name,child_age,child_gender):
        Father.__init__(self,father_name,father_age,father_talents)
        Mother.__init__(self,mother_name,mother_age,mother_talents)
        self.child_name = child_name
        self.child_age = child_age
        self.child_gender = child_gender
        
    def getChildDetails(self):
        self.child_name = input("Enter child name: ")
        self.child_age = int(input("Enter child age: "))
        self.child_gender = input("Enetr the child Gender: ")
        self.father_name = input("Enter Father name: ")
        self.father_age = int(input("Enter Father age: "))
        self.father_talents = input("Enter Father talents: ")
        self.mother_name = input("Enter Mother name: ")
        self.mother_age = int(input("Enter Mother age: "))
        self.mother_talents = input("Enter Mother talents: ")
        
    def printChildDetails(self):
        print("child name is : ",self.child_name)
        print("child age is : ",self.child_age)
        print("child gender is : ",self.child_gender)
        print("Father name is : ",self.father_name)
        print("Father age is : ",self.father_age)
        print("Father Talents:", self.father_talents)
        print("Mother name is : ",self.mother_name)
        print("Mother age is : ",self.mother_age)
        print("Mother Talents:", self.mother_talents)
        
child = Child(None,None,None,None,None,None,None,None,None)
child.getChildDetails()
print("\nChild Details: ")
child.printChildDetails()


# (d) Create a function displayTalents() in class Child that displays the talents of the child inherited from father and mother. A talent is inherited to a child if both the parents possess it.

# In[129]:


class Father:
    def __init__(self,father_name,father_age,father_talents):
        self.father_name = father_name
        self.father_age = father_age
        self.father_talents = [talent.lower() for talent in father_talents]
        
class Mother:
    def __init__(self,mother_name,mother_age,mother_talents):
        self.mother_name = mother_name
        self.mother_age = mother_age
        self.mother_talents = [talent.lower() for talent in mother_talents]
        
class Child(Father,Mother):
    def __init__(self,father_name,father_age,father_talents,mother_name,mother_age,mother_talents,child_name,child_age,child_gender):
        Father.__init__(self,father_name,father_age,father_talents)
        Mother.__init__(self,mother_name,mother_age,mother_talents)
        self.child_name = child_name
        self.child_age = child_age
        self.child_gender = child_gender
        
   
    def displayTalents(self):
        inherited_talents = set(self.father_talents).intersection(set(self.mother_talents))
        if inherited_talents:
            inherited_talents = [talent.capitalize() for talent in inherited_talents]  # Capitalize for better readability
            print("Inherited Talents:", ", ".join(inherited_talents))
        else:
            print("No talents inherited from both parents.")
            
            
father_talents = ["Agriculture", "Gardening"]
mother_talents = ["Cooking", "Gardening" , "agriculture"]

child = Child("Apaprao", 48, father_talents, "Ravanamma", 46, mother_talents, "Venky", 23, "Male")
child.displayTalents()


# # QUESTION-3

# Text File Input Output
# 
#          Create a .txt (text) file and use the pledge of India as the content of the text file.
# 
#          Write a python program that reads this text file, processes it by counting the number of occurrences of each word in the file, and then writes the result back to a          new text file.

# In[130]:


f = open("pledge.txt","w")
f.write("India is my country. All Indians are my brothers and sisters. I love my country and I am proud of its rich and varied heritage. I shall strive to be worthy of it. I shall respect my parents, teachers and all elders and treat everyone with courtesy. To my country and all my people, I pledge my devotion. In their well being and prosperity alone lies my happiness.")
f.close()


# In[131]:


f = open("pledge.txt","r")
data = f.read()
print(data)
print(type(data))
f.close()


# In[132]:


from collections import Counter

def count_words(text):
    words = text.lower().split()
    return Counter(words)

def write_word_count(word_count, output_file):
    with open(output_file, 'w') as file:
        for word, count in word_count.items():
            file.write(f"{word}: {count}\n")

input_file = 'pledge.txt'
output_file = 'word_count.txt'

with open(input_file, 'r') as file:
    text = file.read()

word_count = count_words(text)
write_word_count(word_count, output_file)
print("Word count saved to", output_file)


# In[133]:


f = open("word_count.txt","r")
data = f.read()
print(data)
f.close()


# # QUESTION-4

# For a restaurant, create a parent class ‘Bill’ which has the properties as ‘Customer  name’, ‘Table Number’ and ‘Order’ of which the name, order are strings and table  number is a positive integer. Define a method to extract the order details from the string and return a 2-D array of ordered items & their number. Create a child class ‘ 'Restaurant Bill’ which has a property called ‘Price list’ of the items and has a method to calculate the total bill amount by using the price list and order details. Also have a   method to print the complete bill for the customer including taxes and service charges.
# 
#      The strings will be of the following format:
# 
#      #Name: “Akshay” (Name of the customer)
# 
#      #Table Number: 7 (Table Number)
# 
#      #Order: “Item1x1,Item2x3,Item3x1,…” (ItemxNumber needed)
# 
#      #Price List: “Item1-100,Item2-70,Item3-250,...” (Item-Price)

# In[134]:


class Bill:
    def __init__(self, customer_name, table_number, order):
        self.customer_name = customer_name
        self.table_number = table_number
        self.order = order
    
    def extract_order_details(self):
        order_details = []
        items = self.order.split(',')
        for item in items:
            name, quantity = item.split('x')
            order_details.append([name.strip(), int(quantity)])
        return order_details

class RestaurantBill(Bill):
    def __init__(self, customer_name, table_number, order, price_list):
        super().__init__(customer_name, table_number, order)
        self.price_list = self.create_price_dict(price_list)
    
    def create_price_dict(self, price_list):
        prices = {}
        items = price_list.split(',')
        for item in items:
            name, price = item.split('-')
            prices[name.strip()] = float(price)
        return prices
    
    def calculate_total_bill(self):
        total_bill = 0
        order_details = self.extract_order_details()
        for item, quantity in order_details:
            if item in self.price_list:
                total_bill += self.price_list[item] * quantity
        return total_bill
    
    def print_bill(self):
        total_bill = self.calculate_total_bill()
        tax = 0.1 * total_bill  # Assuming 10% tax
        service_charge = 0.05 * total_bill  # Assuming 5% service charge
        total_amount = total_bill + tax + service_charge

        print("Customer Name:", self.customer_name)
        print("Table Number:", self.table_number)
        print("Order Details:")
        for item, quantity in self.extract_order_details():
            print(f"\t{item}: {quantity}")
        print("Total Bill: $", total_bill)
        print("Tax (10%): $", tax)
        print("Service Charge (5%): $", service_charge)
        print("Total Amount (including tax and service charge): $", total_amount)

order_string = "Item1x1,Item2x2"
price_list_string = "Item1-34,Item2-43"
customer_bill = RestaurantBill("venky", 5, order_string, price_list_string)
customer_bill.print_bill()


# # QUESTION-5

# For the previous question( restaurant Bill) - take name, table no, order details from a file, price list from another file and print the whole bill to the new file.

# In[135]:


class Bill:
    def __init__(self, customer_name, table_number, order):
        self.customer_name = customer_name
        self.table_number = table_number
        self.order = order

    def extract_order_details(self):
        items = self.order.split(',')
        order_details = []
        for item in items:
            item_name, quantity = item.split('x')
            order_details.append([item_name.strip(), int(quantity.strip())])
        return order_details


class RestaurantBill(Bill):
    def __init__(self, customer_name, table_number, order, price_list):
        super().__init__(customer_name, table_number, order)
        self.price_list = self.parse_price_list(price_list)

    def parse_price_list(self, price_list):
        items = price_list.split(',')
        price_dict = {}
        for item in items:
            name, price = item.split('-')
            price_dict[name.strip()] = float(price.strip())
        return price_dict

    def calculate_total_bill(self, tax_rate=0.1, service_charge_rate=0.05):
        order_details = self.extract_order_details()
        total_bill = sum(self.price_list[item] * quantity for item, quantity in order_details)
        total_bill += total_bill * tax_rate
        total_bill += total_bill * service_charge_rate
        return total_bill

    def generate_bill_details(self, tax_rate=0.1, service_charge_rate=0.05):
        bill_details = "#Name: {}\n".format(self.customer_name)
        bill_details += "#Table Number: {}\n".format(self.table_number)
        bill_details += "#Order: {}\n".format(self.order)
        bill_details += "#Price List: {}\n".format(", ".join([f"{item}-{price}" for item, price in self.price_list.items()]))
        bill_details += "Total Bill Amount (including tax and service charge): {}\n".format(self.calculate_total_bill(tax_rate, service_charge_rate))
        return bill_details


def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)


# Create sample data and write to files
customer_name = "venky"
table_number = "5"
order = "Item1x1,Item2x2"
price_list = "Item1-34,Item2-43"

write_to_file('customer_name.txt', customer_name)
write_to_file('table_number.txt', table_number)
write_to_file('order_details.txt', order)
write_to_file('price_list.txt', price_list)

# Read data from files
customer_name = open('customer_name.txt').read().strip()
table_number = int(open('table_number.txt').read().strip())
order = open('order_details.txt').read().strip()
price_list = open('price_list.txt').read().strip()

# Create bill object
bill = RestaurantBill(customer_name, table_number, order, price_list)

# Generate bill details
bill_details = bill.generate_bill_details()

# Write bill details to file
write_to_file('bill_details.txt', bill_details)

print("Bill details have been written to 'bill_details.txt'")


# In[136]:


f = open("bill_details.txt","r")
data = f.read()
print(data)
f.close()

