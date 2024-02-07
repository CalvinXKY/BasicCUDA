

# The c++ .so pkg is needed before importing.
import classes

"""
The snippet shows how to invoke a bond class in python, as below:
"""

# Create obj from c++ class.
shape = classes.Shape(1, 10, "Basic example of c++ class")
print(shape)

# Change the area while invoke "setProperty" func with 'float' tpye input.
shape.setProperty(20.0)

# "setProperty" func has been overloaded, thus it can also change the num.
shape.setProperty(2)

# Then we check the info:
print(shape)

# It could raise error while call a function not defined in "PYBIND11_MODULE"
# shape.getProfile()

"""
Inheritance example:
"""

rect = classes.Rectangle(1, 10, 10, "Inheritance example.")
print(rect.getArea())

# Profile attribution could be read&write:
rect.profile = "The profile description has been changed!"
print(rect)
