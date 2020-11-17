#!/usr/bin/python3
import sys

# ## Loop over the text inputs from stdin. (NOTE: this is not a function - just the raw code)
# Find out how to access stdin from the sys module
# Lowercase the text, and split it on whitespace.
#
# We are going to write/output every word to stdout in a TSV format
# where 1 is the value.
#
# Example input: "Lorem Ipsum...? I don't know the rest of it"
# Example output: lorem 1
#                 ipsum...? 1
#                 i 1
#                 etc.
for text_string in sys.stdin:
    text_string = text_string.lower().split()
    for word in text_string:
        print("{}\t{}".format(word, 1))
