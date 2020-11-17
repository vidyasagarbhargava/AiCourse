#!/usr/bin/python3
import sys

# Loop over the lines from stdin, and sum up the values for all the unique keys
# NOTE: Again, this is not a function - enter your code directly.
# NOTE: Assume that the inputs are sorted for you
# We're going to output things to stdout again.
# Our output will TSVs, where the word is the key and the total number of occurances of that word is the value
# Example input: a  1
#                a  1
#                a  1
#                ....
#                core   1
#                core   1
#                etc.
# Example output: a 3
#                 core  2
