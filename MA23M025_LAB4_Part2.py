#!/usr/bin/env python
# coding: utf-8

# # QUESTION-6

# Store the employee IDs, names, salaries, and years of experience using nested dictionaries (the key of the highest level dictionary can be the employee ID). 
# 
# a) Sort this dictionary using the salary value. 
# 
# b) Add a new employee to the dictionary in the correct position (sorted as mentioned above).

# In[37]:


# Define the initial nested dictionary with employee information
employees = {
    25: {'name': 'venky', 'salary': 80000, 'experience': 1},
    26: {'name': 'vinod', 'salary': 90000, 'experience': 2},
}

# Sort the dictionary based on salary and print
sorted_employees = dict(sorted(employees.items(), key=lambda x: x[1]['salary']))
print("Sorted employees by salary :",sorted_employees)

# Adding new employee
new_employee = {'name': 'hari', 'salary': 95000, 'experience': 3}

# Finding  correct position to insert the new employee based on salary and insert
insert_position = next((index for index, (emp_id, emp_info) in enumerate(sorted_employees.items()) if emp_info['salary'] > new_employee['salary']), len(sorted_employees))
sorted_employees = dict(list(sorted_employees.items())[:insert_position] + [(27, new_employee)] + list(sorted_employees.items())[insert_position:])

# updated dictionary
print("\nUpdated employees with new employee :",sorted_employees)


# # QUESTION-7

# You are given two Python dictionaries, A and B, with keys as alphabets and values as random integers. Write a Python function to create a third dictionary C, that combines A and B. For common keys, the value in C should be the sum of values from A and B. 
# 
# For example, if dictionary A is {"a": 3, "b": 5, "c": 7} and dictionary B is {"b": 2, "c": 4, "d": 6}, the function should return a dictionary C that looks like {"a": 3, "b": 7, "c": 11, "d": 6}.

# In[38]:


def combine_dicts(A, B):
    C = {}
    for key in set(A.keys()) | set(B.keys()):
        C[key] = A.get(key, 0) + B.get(key, 0)
    return C

# Example dictionaries A and B
A = {"a": 3, "b": 5, "c": 7}
B = {"b": 2, "c": 4, "d": 6}

#combined dictionary is 
C = combine_dicts(A, B)
print(C)  


# # QUESTION-8

# Assume you have a list of lists, where each inner list contains two elements: a key and a value. Write a Python function that takes the list of lists as input and returns a list of dictionaries, where each dictionary contains a key-value pair from the original input list.

# In[39]:


def list_of_lists_to_dicts(list_of_lists):
    list_of_dicts = []
    for inner_list in list_of_lists:
        if len(inner_list) == 2:  # Ensure each inner list contains exactly two elements
            key, value = inner_list
            list_of_dicts.append({key: value})
    return list_of_dicts

# Example list of lists
list_of_lists = [['a', 1], ['b', 2], ['c', 3]]

# Print the list of dictionaries
list_of_dicts = list_of_lists_to_dicts(list_of_lists)
print(list_of_dicts) 


# # QUESTION-9

# Illustrate the usage of positional and keyword arguments using suitable examples.

# In[40]:


#positional arguments
def details(name,rollnumber):
    print(f"This is {name} and my rollnumber is {rollnumber}")

details("venkatesh",12345)


# In[41]:


#keyword arguments
details(name='vinod',rollnumber=345)
#combination of both positional and key word arguments
details('Suman', rollnumber= 3452)


# # QUESTION-10

#  Write a function to find the maximum of n numbers using variable length positional arguments.

# In[42]:


def maxi(*numbers):
    if not numbers:
        return None
    return max(numbers)

# Example usage:
result = maxi(1,4,5,3,4,5)
print("Maximum number:", result)  


# # QUESTION-11

# Write a function to concatenate n strings using variable length keyword arguments.

# In[43]:


def concatenate_strings(**kwargs):
    return ''.join(kwargs.values())

# Example :
result = concatenate_strings(a="This", b=" ", c="is", d=" ", e="ED5340", f=" ", g=" course")
print("Concatenated string:", result)  

