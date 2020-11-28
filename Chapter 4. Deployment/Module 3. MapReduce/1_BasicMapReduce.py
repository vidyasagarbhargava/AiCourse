# %%
import os
from itertools import groupby

# %%
GUTENBERG_DIR = "../DATA/gutenberg/"
gutenberg_books = []
# ## Load the books into gutenberg_books
# You may need to declare the file encodings as 'utf-8'


# %%
print(len(gutenberg_books))

# Concatenate all the books together into a string
gutenberg_books = " ".join(gutenberg_books)

print(len(gutenberg_books))
# %%
# ## Define a map function.
# This map function will take a string as an argument, and output a list of tuples. Each tuple will be a (K,V) pair where K is a word, and V is 1
# Note, this is where we would also perform some pre-processing.
# This is a MR tutorial though, not an NLP one, so the only preprocessing we will apply is lowercasing the text!


def map(books: str):
    pass


# Can you think of how we could augment our pre-processing?
mapped = map(gutenberg_books)
print(mapped[:10])

# %%
# ## Define a (sort and) shuffle function
# Sort the list on K, and then shuffle the list. In the shuffle step, I want a dictionary as output. The dictionary will contain {K, V[]} pairs. Again, K is a word, but V[] is a list of values. This list should be filled with as many 1s as there are instances of the K in the sorted list.
# Hint, you may find the groupby function useful here


def shuffle(list_of_tuples):
    pass


shuffled = shuffle(mapped)
print(shuffled["blue"])

# %%
# ## Define and run a reduce function
# The reduce function will receive a dictionary of {K, V[]} pairs and reduce the output to (K, V), where K is the word/key, and V is the sum (count would also work in this case) of the V[] elements


def reduce(shuffled_output):
    pass


reduced = reduce(shuffled)
print(reduced)
