

import functions

def divide_print(info):
    print("\n", "-"*40, "\n", "-"*5, info, "\n", "-"*40)

# Inference will not work
divide_print(" Add case")
print("4+3=", functions.add_two_num(4, 3))
print("With default args, the result is:", functions.add_two_num_with_default())

# Functions overload, the last one works:
divide_print(" overlaod case")
functions.printInfo(10)


# Certain basic Python types (like str, int, bool, float, etc.) are immutable. See "Limitations involving reference arguments"
# Inference will not work
divide_print(" Inplace case")
num = 10
print("Before inplace opt, num: ", num)
functions.inplace_add(num, 10)
print("After inplace opt, num: ", num)

# Sturct data type is OK while using inplace operation
divide_print("Inplace  case (corrected):")
data = functions.Data()
data.num = 10
print("Before inplace opt, data.num: ", data.num)
functions.inplace_add_use_struct(data, 4)
print("After inplace opt, data.num: ",data.num)

# data pointer
divide_print("Function with struct ptr variable in c++ called in python")
data.num = 0
functions.set_data_ptr_100(data)
print(data.num)

# Call a variable:
# Sturct data type is OK while using inplace operation
divide_print("Global variable:")
print("Print the variable:",functions.worldCount)
functions.worldCount = 2
print("Change the variable:",functions.worldCount)

# Template:
divide_print("Template: multiply(T, T)")
print("int * int:", functions.multiply(2, 3))
print("float * float:", functions.multiply(2.0, 3.0))

# Explicit args, no convert:
print("float * float (not allow convert):")

try:
    functions.multiply_float(1,3)
except TypeError as e:
    print("TypeError Case: \n", e)

# Allow/Prohibiting None arguments
divide_print("Allow/Prohibiting None arguments")
functions.show_data_num(data)
try:
    # Run with None, will raise an error:
    functions.show_data_num(None)
except TypeError as e:
    print("TypeError Case: \n", e)
functions.show_data_num_allow_none(None) # That's ok.

# Recall function

