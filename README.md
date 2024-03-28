# LLM-Tokenization

This repository is inspired by the Andrej Karpathy lecture below as well as his repository addressing the creation of `minbpe`, a minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization:

[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=95s)
|---|

[minbpe Github repository](https://github.com/karpathy/minbpe/tree/master)
|---|

The main intention of this project is - before referencing any already created Tokenizers (i.e. [tiktoken](https://github.com/openai/tiktoken) or [minbpe](https://github.com/karpathy/minbpe/tree/master)) in LLM models - to understand how they work behind the scenes and its importance.

## Motivation

It’s hard to find free online resources (at least that's my perception), on what a LLM tokenizer is, why it is needed and how it can be created from scratch using code.

I would’ve loved to find some kind of resource that explains the same in a structured text format and sections, therefore, I’m creating this repository. 

Not only as a knowledge record so I can inspect in the future, but with the hopes that someone can find it useful, instead of referencing tokenizers into their models acting like a black box (not understanding what it does, hence, unable to know why a model could misbehave).

## Introduction // Why do we need Tokenization?

Why do we need tokenization? 

To train LLM with data, that being text, we need to format such data so the model (and the resources behind) is able to 'understand it' and find patterns.

The easiest way is to convert our text data into tokens. But... What do you mean by tokens?

Tokens will be our way to convert the text data into something the machine can find patterns. 

Something like teaching a kid how to speak. You don't give a kid that doesn't know how to read a book and expect to find patterns on the words to understand it.

You must teach him the alphabet first, then, the combination of the letters of the alphabet can form words. You have to explain what a word is, that being a textual (in a LLM context) representation of 'something'.

Words can form sentences, which are just a specific sequential concatenation of words that have a meaning based on the context/meaning of the word.

It's like our brain creates some sort of an embedded table for each character, word and sentence we know, so we can establish patterns when we read text, reinforced by the text data we ingest during our lifetime. The older you get, more information you ingest, more patterns to analyze, the wiser you get.

We must follow the same process to teach our model or computational resource, whatever you want to call it, how to find patterns in text.

But, how can a computer know what "hello" means?

A kid will know what "hello" means by associating the word with a context and observing its environment:

=== ENVIRONMENT ===

Situation: A person approaches another.

- "Hello, how are you?"

After repetition of the same, the kid will know that the word "hello" is associated to the observation of a person interacting with another when both get close enough, and some time has passed (i.e. you don't say "hello" each time you see your mom in your house in the time context of a day)

A kid will figure out what "hello" means, and when the kid sees a person, will say "hello".

But, a computer cannot "see" or observe the world in the same way we do, so how can we teach it?

That's how the tokenization begins. It's our way to dissect text data into smaller groups, and find similarities between the groups given the corresponding context of the provided text data.

For example, let's provide the following the text into the model:

```
"Hello, are you happy? I've seen you smiling from far!"
```

What patterns are in that string? Let's dissect it and split the string into sequential characters (sort of teaching the alphabet):

```
['H', 'e', 'l', 'l', 'o', ',', ' ', 'a', 'r', 'e', ' ', 'y', 'o', 'u', ' ', 'h', 'a', 'p', 'p', 'y', '?', ' ', 'I', "'", 'v', 'e', ' ', 's', 'e', 'e', 'n', ' ', 'y', 'o', 'u', ' ', 's', 'm', 'i', 'l', 'i', 'n', 'g', ' ', 'f', 'r', 'o', 'm', ' ', 'f', 'a', 'r', '!']
```

We can observe that when “H” happens, “e” comes next.

When “H” and “e” happen, “l” comes next.

And so on. If we imagine a very long text where we apply this logic, the model will be able to statistically know what character comes next in the context of the previous character, and the previous context of the previous character, so on so forth.

However, machines work better with numbers. Why can't we associate each character to a number, so the machine has an embedded table of the same? That’s called `vectorization`. Example:

```
Word “Hello”

== encoding ==

“H” “1”
“e” “2”
“l” “3”
“o” “4”

Machine dissection of the word hello:

Tokenization → ['H', 'e', 'l', 'l', 'o']

Vectorization → ['1', 2', '3', '3', '4']
```

The machine will understand that when a 1 happens the 2 comes next, and so on.

That will work under certain text data, but it’s inefficient. We must correlate a number to every character existing in the text data. That will show a long list of vocabulary patterns if the text data contains text in several languages. Therefore, if the text provided to the model is too rich in different characters, the processing will be too large (one number to each character).

So… Is there a way to reduce that computationally expensive action similarly to what we do when we humans read words instead of individual characters?

Yes, what we can do is find repetitive patterns to encode our text data to make it more computationally efficient.

Let’s analyze the following:

```
Word “Honolulu”

== encoding ==

“H” “1”
“o” “2”
“n” “3”
“l” “4”
“u” “5”

Machine dissection of the word “Honolulu”:

Tokenization → ['H', 'o', 'n', 'o', 'l', 'u', 'l', 'u']

Vectorization → ['1', 2', '3', '2', '4', ‘5’, ‘4’, ‘5’]
```

We can observe that in “Honolulu”, ‘lu’ is being referenced two times, let’s encode it:

```
Word “Honolulu”

== encoding ==

“H” “1”
“o” “2”
“n” “3”
“l” “4”
“u” “5”
“lu” “6”

Machine dissection of the encoded word “Honolulu”:

Tokenization → ['H', 'o', 'n', 'o', 'l', 'u', 'l', 'u']

Vectorization → ['1', 2', '3', '2', '6’, ‘6’]
```

This is just a simple word, imagine a large text dataset with several occurrences being capable of being encoded, it will reduce computation a lot!

Now, we have an understanding on how machines are capable of establishing patterns in text efficiently, but, how can we transfer all of this explanation into actual code?

Below we will proceed to do so.

## How to tokenize, vectorize, encode and decode our text data with code (Python)

Now we know what tokenizing, vectorizing and encoding is in LLM context, which involves breaking down text data into smaller, meaningful units (tokens) such as words, characters, or sub-words, which can then be processed by machines more efficiently.

There are a few ways to tokenize and vectorize data. As we saw, we can use character level tokenization and vectorization (inefficient), or as this project tries to show, we can encode our data as well in repetitive chunks for better computational processing.

We will delve into the creation of a tokenizer using the BPE algorithm.

[What is BPE (Byte pair encoding)?](https://en.wikipedia.org/wiki/Byte_pair_encoding)
|---|

The BPE algorithm is just the code expression of the “Honolulu'' example. So let's start.

We will use Python, and the text data that we will process will be `str` objects. 

[What is a str object in Python](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)
|---|

In the link, you will be able to see that strings are immutable sequences of Unicode code points, but what are Unicode code points?

[What are Unicode code points?](https://en.wikipedia.org/wiki/Unicode)
|---|

Unicode code points are unique numbers assigned to represent characters in a system for representing text. These code points are expressed in the form "U+1234", where "1234" is the assigned number. For example, the character "A" is assigned a code point of U+0041. Character encoding forms, such as UTF-8, determine how a Unicode code point should be encoded as a sequence of bytes.

[What is UTF-8?](https://en.wikipedia.org/wiki/UTF-8)
|---|

UTF-16 and UTF-32 also exist, however, the optimal one for LLM context is UTF-8. So, let’s stick to it.

But… How is all of that expressed into code? For the sake of simplicity let's call `tokens` to the vectorized (numeral) representation of a string. 

For example the string "Hello world", will be the sequence of the following unicode points (Feel free to paste that code into a [Google colab notebook](https://colab.research.google.com/) and execute it):

```
# Original string
text = "Hello world"

# Tokenizing each character
tokens = list(text.encode("utf-8"))

# Printing the tokens
print("unicode points in text: ", tokens)

unicode points in text:  [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
```

As you can see, each character is being associated with a numerical representation. That will allow us to create a character level vocabulary tokenizer, however, we will use BPE as indicated to find patterns.

How can we find such patterns with code? Let's use the “Honolulu” example:

```
text = "Honolulu"
tokens = list(text.encode("utf-8"))

def getstats(ids):
  counts = {}
  for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
  return counts

stats = getstats(tokens)
print("patterns in the text: ", stats)

patterns in the text:  {(72, 111): 1, (111, 110): 1, (110, 111): 1, (111, 108): 1, (108, 117): 2, (117, 108): 1}
```

We see that the pattern (108, 117) occurs twice, which stands for ‘l’ ‘u’:

```
chr(108), chr(117)

('l', 'u')
```

Well… Obviously we are not going to train a LLM with just a word. We will feed the model with extensive amounts of text, therefore, we must take into account some concepts.

As we are encoding using `UTF-8` the data is returned in bytes format, therefore, from 0 to 255 bytes. That means that we need to expand that range in order to accommodate the amount of tuples that we found necessary in order to encode further our text for better computing.

In our “Honolulu” case, we will accommodate an extra tuple. Since the default byte range is `0 to 255`, let’s define that our new vocabulary will be of 257 bytes, the 256 byte (the last one) being the tuple that we found in the word.

In order to achieve that, we can use the following code, which will return the new encoded representation of the word “Honolulu”:

In order to achieve that, we can use the following code, which will merge the repeated tuple into our new byte, and substitute the previous encoded output:

```
def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# ---
vocab_size = 257 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = getstats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

print("Previous encoding: ", tokens, "New encoding: ", ids)

merging (108, 117) into a new token 256
Previous encoding:  [72, 111, 110, 111, 108, 117, 108, 117] New encoding:  [72, 111, 110, 111, 256, 256]
```

We can observe that our word has been successfully encoded.

As we observed with the “Honolulu” example and the 257 byte, we can increase the amount of vocabulary given a text to ensure that the amount of tuples that we found necessary can be merged and encoded further to reduce the amount of tokens created.

If we increase the vocabulary of a given text, that is called `training` the tokenizer. The amount of training (increasing the amount of vocabulary) you can perform will depend on the text provided, as well as the performance of the LLM model trained with that text.

So, this project just tries to show how tokenizers work, and the training will fall outside of this. We just need to know that we can train a tokenizer and increase the amount of vocabulary if needed. 

Coming back to the “Honolulu” example, the code shown will just work for that word since it is hard coded. We must create a function that will encode text based on whatever input we introduce, as well as encoding that text based on a given vocabulary.

Moreover, we are just encoding! Imagine trying to understand encoded text data. We will not be able to understand it if we don't decode back to text so humans can understand.

Mmm, we then need to create these 3 functions:

1.- A function to `train` the tokenizer. That means, given our text and a vocabulary integer >= 256, a table with extra repetitive tuples must be created so the encode and decode function can perform lookups.

2.- A function to `encode` our text based on the trained table generated on the `train` function.

3.- A function to `decode` the vectors generated by the `encode` function into plain text.

In order to achieve this, we must create a basic Class with Python to introduce our functions. The code for the same is below (link to the code generated by Andrej Karpathy):

[BasicTokenizer](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py)
|---|

But, how can we run and check that code? Feel free to paste that code into a [Google colab notebook](https://colab.research.google.com/) and execute it.

To run successfully, you must paste these two functions before the BasicTokenizer Class, and remove the line `from .base import Tokenizer, get_stats, merge` of the BasicTokenizer Class file:

```python
def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids
```

With the following code, we will be able to train, encode and decode our text:

```python
# Create an instance of the BasicTokenizer class
t = AutoTokenizer()

# Train the tokenizer with some text and a vocabulary size
text = "honolulu"
vocab_size = 257
t.train(text, vocab_size)

# Encode a text string
e = t.encode(text)

# Decode the vectors
d = t.decode(t.encode(text))

# Verify if the encoded and consequent decoding is the same as the original text, to avoid loss of the text source truth.
v = t.decode(t.encode(text)) == text

print("Original text: ", text)
print("Encoded text: ", e)
print("Decoded text: ", d)
print("Is the decoding of the encoding the same indicating that no data was lost? True or False?", v)
```

Output:

```
Original text:  honolulu
Encoded text:  [104, 111, 110, 111, 256, 256]
Decoded text:  honolulu
Is the decoding of the encoding the same indicating that no data was lost? True or False? True
```

Now you should be able to play changing the text as well as the vocabulary to observe how the tokens are generated.

Furthermore, you now understand the basics concepts of how character level tokenization (inefficient) and BPE (efficient) tokenization works.

## What's next?

This is a brief introduction on character level and BPE style LLM tokenizers. There’s a lot of ways to modify such tokenizers and add functionalities to them.

That will be another repository to delve further.

## Bibliography

https://www.datacamp.com/blog/what-is-tokenization 

https://github.com/karpathy/minbpe/tree/master

https://github.com/openai/tiktoken

https://en.wikipedia.org/wiki/Byte_pair_encoding

https://colab.research.google.com/

https://medium.com/@WojtekFulmyk/text-tokenization-and-vectorization-in-nlp-ac5e3eb35b85 

https://en.wikipedia.org/wiki/Unicode 

https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str 

https://www.youtube.com/watch?v=zduSFxRajkE&t=95s 

https://en.wikipedia.org/wiki/UTF-8

https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py 
