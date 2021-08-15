Creating and Visualizing Word Vectors Using Word2Vec
-------------------------------------------------------------------------

In this lab, we will cover the following recipes:


-   Acquiring data
-   Importing the necessary libraries
-   Preparing the data
-   Building and training the model
-   Visualizing further
-   Analyzing further

Introduction
------------------------------

Before training a neural network on text data
and generating text using LSTM cells, it is important to understand how
text data (such as words, sentences, customer reviews, or stories) is
converted to word vectors first before it is fed into a neural network.
This lab will describe how to convert a text into a corpus and
generate word vectors from the corpus, which makes it easy to group
similar words using techniques such as Euclidean distance calculation or
cosine distance calculation between different word vectors.


Acquiring data
--------------------------------

The first step is to acquire some data to
work with. For this lab, we will require a lot of text data to
convert it into tokens and visualize it to understand how neural
networks rank word vectors based on Euclidean and Cosine distances. It
is an important step in understanding how different words get associated
with each other. This, in turn, can be used to design better, more
efficient language and text-processing models.


### Getting ready

Consider the following:


-   The text data for the model needs to be in files of `.txt`
    format, and you must ensure that the files are placed in the current
    working directory. The text data can be anything from Twitter feeds,
    news feeds, customer reviews, computer code, or whole books saved in
    the `.txt` format in the working directory. In our case,
    we have used the [*Game of Thrones*] books as the input
    text to our model. However, any text can be substituted in place of
    the books, and the same model will work.
-   Many classical texts are no longer protected under copyright. This
    means that you can download all of the text for these books for free
    and use them in experiments, such as creating generative models. The
    best place to get access to free books that are no longer protected
    by copyright is Project
    Gutenberg (<https://www.gutenberg.org/>).


### How to do it\...

The steps are as follows:

1.  Begin by visiting the Project Gutenberg website and browsing for a
    course that interests you. Click on the course, and then click on
    **`UTF-8`**, which allows you to download the course in plain-text
    format. The link is shown in the following screenshot:


![](./images/ddd68374-24bd-461d-86a8-d7fa8ec51b01.png)


Project Gutenberg Dataset download page

2.  After clicking on**`Plain Text UTF-8`**, you should see a page that
    looks like the following screenshot. Right click on the page and
    click on**`Save As..`**. Next, rename the file to whatever you
    choose and save it in your working directory:


![](./images/94984046-5f6e-49ed-afe8-9fa034c3fd05.png)


3.  You should now see a `.txt` file with the specified
    filename in your current working directory.
4.  Project Gutenberg adds a standard header and footer to each course;
    this is not part of the original text. Open the file in a text
    editor, and delete the header and the footer.


### How it works\...

The functionality is as follows:

1.  Check for the current working directory using the following
    command: `pwd`.
2.  The working directory can be changed
    using the `cd` command as shown in the following
    screenshot:


![](./images/3bc0b467-f49f-4ed8-b6c0-94f17810ff48.png)


3.  Notice that, in our case, the text files are contained in a folder
    named `USF`, and, therefore, this is set as the working
    directory. You may similarly store one or more `.txt`
    files in the working directory for use as input to the model.
4.  UTF-8 specifies the type of encoding of the characters in the text
    file. **UTF-8** stands for **Unicode Transformation
    Format**. The **8** means it uses
    **8-bit** blocks to represent a character.
5.  UTF-8 is a compromise character encoding that can be as compact as
    ASCII (if the file is just plain-English text) but can also contain
    any Unicode characters (with some increase in file size).
6.  It is not necessary for the text file to be in a UTF-8 format, as we
    will use the codecs library at a later stage to encode all the text
    into the Latin1 encoding format.


### There\'s more\...

For more information about UTF-8 and Latin1 encoding formats, visit the following
links:


-   <https://en.wikipedia.org/wiki/UTF-8>
-   <http://www.ic.unicamp.br/~stolfi/EXPORT/www/ISO-8859-1-Encoding.html>


### See also

Visit the following link to understand the need for word vectors in
neural networks
better:<https://medium.com/deep-math-machine-learning-ai/lab-9-1-nlp-word-vectors-d51bff9628c1>

Listed below are some other useful articles related to the topic of
converting words to
vectors:<https://monkeylearn.com/blog/word-embeddings-transform-text-numbers/>

<https://towardsdatascience.com/word-to-vectors-natural-language-processing-b253dd0b0817>


Importing the necessary libraries
---------------------------------------------------

Before we begin, we require the following
libraries and dependencies, which need to be imported into our Python
environment. These libraries will make our tasks a lot easier, as they
have readily available functions and models that can be used instead of
doing that ourselves. This also makes the code more compact and
readable.


### Getting ready

The following libraries and dependencies will be required to create word
vectors and plots and visualize the n-dimensional word vectors in a 2D
space:


-   `future`
-   `codecs`
-   `glob`
-   `multiprocessing`
-   `os`
-   `pprint`
-   `re`
-   `nltk`
-   `Word2Vec`
-   `sklearn`
-   `numpy`
-   `matplotlib`
-   `pandas`
-   `seaborn`


### How to do it\...

The steps are as follows:

1.  Type the following commands into your
    Jupyter notebook to import all the required libraries:

```
from __future__ import absolute_import, division, print_function
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy
as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%pylab inline
```


2.  You should see an output that looks like the following screenshot:


![](./images/46b3efc0-cfeb-4e13-a03c-655ba509e165.png)


3.  Next, import the `stopwords` and `punkt`
    libraries using the following commands:

```
nltk.download("punkt")
nltk.download("stopwords")
```


4.  The output you see must look like the following screenshot:


![](./images/899e600b-b3d0-432b-85cb-2a98e6c1a39a.png)


### How it works\...

This section will describe the purpose of
each library being used for this recipe.

1.  The `future` library is the missing link between Python 2
    and Python 3. It acts as a bridge between the two versions and
    allows us to use syntax from both versions.

2.  The `codecs` library will be used to perform the encoding
    of all words present in the text file. This constitutes our dataset.

3.  Regex is the library used to look up or search for a file really
    quickly. The `glob` function allows quick and efficient
    searching through a large database for a required file.

4.  The `multiprocessing` library allows us to perform
    concurrency, which is a way of running multiple threads and having
    each thread run a different process. It is a way of making programs
    run faster by parallelization.                                     
          

5.  The `os` library allows easy interaction with the
    operating system, such as a Mac, Windows, and so on, and performs
    functions such as reading a file.

6.  The `pprint` library provides a capability
    for pretty-printing arbitrary Python data structures in a form that
    can be used as input to the interpreter.

7.  The `re` module provides regular expression matching
    operations similar to those found in Perl.

8.  NLTK is a natural language toolkit capable of tokenizing words in
    very short code. When fed in a whole sentence, the
    `nltk` function breaks up sentences and outputs tokens for
    each word. Based on these tokens, the words may be organized into
    different categories. NLTK does this by comparing each word with a
    huge database of pre-trained words called
    a **lexicon**.

9.  `Word2Vec` is Google\'s model, trained on a huge dataset
    of word vectors. It groups semantically similar words close to one
    another. This will be the most important library for this section.

10. `sklearn.manifold` allows the dimensionality reduction of
    the dataset by employing **t-distributed
    Stochastic Neighbor Embedding** (**t-SNE**)
    techniques. Since each word vector is multi-dimensional, we require
    some form of dimensionality reduction techniques to bring the
    dimensionality of these words down to a lower dimensional space so
    it can be visualized in a 2D space.


### There\'s more\...

`Numpy` is a commonly used `math` library.
`Matplotlib` is the `plotting` library we will
utilize, and `pandas` provide a lot of flexibility in data
handling by allowing easy reshaping, slicing, indexing, subsetting, and
manipulation of data.

The `Seaborn` library is another statistical data visualization library that we require along with
`matplotlib`. `Punkt` and `Stopwords` are
two data-processing libraries that simplify tasks such as splitting a
piece of text from a corpus into tokens (that is, via tokenization) and
removing `stopwords`. 

### See also

For more information regarding some of the libraries utilized, visit the following links:


-   [https://docs.python.org/3/library/codecs.html](https://docs.python.org/3/library/codecs.html)
-   <https://docs.python.org/2/library/pprint.html>
-   <https://docs.python.org/3/library/re.html>
-   <https://www.nltk.org/>
-   <https://www.tensorflow.org/tutorials/word2vec>
-   <http://scikit-learn.org/stable/modules/manifold.html>


Preparing the data
------------------------------------

A number of data-preprocessing steps are to be performed before the data
is fed into the model. This section will describe how to clean the
data and prepare it so it can be fed into the
model.


### Getting ready

All the text from the `.txt` files is first converted into one
big corpus. This is done by reading each sentence from each file and
adding it to an empty corpus. A number of preprocessing steps are then
executed to remove irregularities such as white spaces, spelling errors,
`stopwords`, and so on. The cleaned text data has to then be
tokenized, and the tokenized sentences are added to an empty array by
running them through a loop.

### How to do it\...

The steps are as follows:

1.  Type in the following commands to search for the `.txt`
    files within the working directory and print the names of the files
    found:

```
book_names = sorted(glob.glob("./*.txt"))
print("Found books:")
book_names
```


In our case, there are five books named `got1`,
`got2`, `got3`, `got4`, and
`got5` saved in the working directory. 

2.  Create a `corpus`, read each sentence starting with the
    first file, encode it, and add the encoded characters to a
    `corpus` using the following commands:

```
corpus = u''
for book_name in book_names:
print("Reading '{0}'...".format(book_name))
with codecs.open(book_name,"r","Latin1") as book_file:
corpus += book_file.read()
print("Corpus is now {0} characters long".format(len(corpus)))
print()
```


3.  Execute the code in the preceding steps, which should result in an
    output that looks like the following screenshot:


![](./images/280a7a97-083f-4175-aaef-ca2fb4ceee23.png)


4.  Load the English pickle `tokenizer` from `punkt`
    using the following command:

```
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
```


5.  `Tokenize` the entire `corpus` into sentences
    using the following command:

```
raw_sentences = tokenizer.tokenize(corpus)
```


6.  Define the function to split sentences
    into their constituent words as well as remove unnecessary
    characters in the following manner:

```
def sentence_to_wordlist(raw):
     clean = re.sub("[^a-zA-Z]"," ", raw)
     words = clean.split()
     return words
```


7.  Add all the raw sentences where each word of the sentence is
    tokenized to a new array of sentences. This is done by using the
    following code:

```
sentences = []
for raw_sentence in raw_sentences:
  if len(raw_sentence) > 0:
  sentences.append(sentence_to_wordlist(raw_sentence))
```


8.  Print a random sentence from the corpus to visually see how the
    `tokenizer` splits sentences and creates a word list from
    the result. This is done using the following commands:

```
print(raw_sentences[50])
print(sentence_to_wordlist(raw_sentences[50]))
```


9.  Count the total tokens from the dataset using the following
    commands:

```
token_count = sum([len(sentence) for sentence in sentences])
print("The course corpus contains {0:,} tokens".format(token_count))
```


### How it works\...

Executing the tokenizer and tokenizing all
the sentences in the corpus should result in an output that looks like
the one in the following screenshot:


![](./images/0408f3be-a75c-472b-986b-b1afafddda43.png)


Next, removing unnecessary characters, such as hyphens and special
characters, are done in the following manner. Splitting up all the
sentences using the user-defined `sentence_to_wordlist()`
function produces an output as shown in the following screenshot:


![](./images/50b85bcc-c7ea-48b9-92f3-6433a4b47362.png)


Adding the raw sentences to a new array named `sentences[]`
produces an output as shown in the following screenshot:


![](./images/e7d96ae9-d7f5-4c3e-86cb-69d72f05f680.png)


On printing the total number of tokens in the corpus, we notice that
there are 1,110,288 tokens in the entire corpus. This is illustrated in
the following screenshot:


![](./images/46d0300b-15f1-41a3-a9ac-67a101af725f.png)


The functionality is as follows:

1.  The pre-trained `tokenizer` from NLTK is used to tokenize
    the entire corpus by counting each sentence as a token. Every
    tokenized sentence is added to the variable
    `raw_sentences`, which stores the tokenized sentences.
2.  In the next step, common stopwords are removed, and the text is
    cleaned by splitting each sentence into its words.
3.  A random sentence along with its wordlist is printed to understand
    how this works. In our case, we have chosen to print the 50th
    sentence in the `raw_sentences` array.
4.  The total number of tokens (in our case, sentences) in the sentences
    array are counted and printed. In our case, we see that 1,110,288
    tokens are created by the `tokenizer`.


### There\'s more\...

More information about tokenizing paragraphs and sentences can be found by visiting the following links:


-   <https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize>
-   <https://stackoverflow.com/questions/37605710/tokenize-a-paragraph-into-sentence-and-then-into-words-in-nltk>
-   <https://pythonspot.com/tokenizing-words-and-sentences-with-nltk/>


### See also

For more information about how regular
expressions work, visit the following link:

<https://stackoverflow.com/questions/13090806/clean-line-of-punctuation-and-split-into-words-python>


Building and training the model
-------------------------------------------------

Once we have the text data in the form of
tokens in an array, we are able to input it
in the array format to the model. First, we have to define a number of
hyperparameters for the model. This section will describe how to do the
following:


-   Declare model hyperparameters
-   Build a model using `Word2Vec`
-   Train the model on the prepared dataset
-   Save and checkpoint the trained model

### Getting ready

Some of the model hyperparameters that are to be declared include the
following:


-   Dimensionality of resulting word vectors
-   Minimum word count threshold
-   Number of parallel threads to run while training the model
-   Context window length
-   Downsampling (for frequently occurring words)
-   Setting a seed


Once the previously mentioned hyperparameters are declared, the model
can be built using the `Word2Vec` function from the
`Gensim` library.

### How to do it\...

The steps are as follows:

1.  Declare the hyperparameters for the model using the following
    commands:

```
num_features = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1
```


2.  Build the model, using the declared hyperparameters, with the
    following lines of code:

```
got2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
```


3.  Build the model\'s vocabulary using the tokenized sentences and
    iterating through all the tokens. This is done using the
    `build_vocab` function in the following manner:

```
got2vec.build_vocab(sentences,progress_per=10000, keep_raw_vocab=False, trim_rule=None)
```


4.  Train the model using the following command:

```
got2vec.train(sentences, total_examples=got2vec.corpus_count, total_words=None, epochs=got2vec.iter, start_alpha=None, end_alpha=None, word_count=0, queue_factor=2, report_delay=1.0, compute_loss=False)
```


5.  Create a directory named trained, if it doesn\'t already exist. Save
    and checkpoint the `trained` model using the following
    commands:

```
if not os.path.exists("trained"):
     os.makedirs("trained")
got2vec.wv.save(os.path.join("trained", "got2vec.w2v"), ignore=[])
```


6.  To load the saved model at any point, use the following command:

```
got2vec = w2v.KeyedVectors.load(os.path.join("trained", "got2vec.w2v"))
```


### How it works\...

The functionality is as follows:

1.  The declaration of model parameters does
    not produce any output. It just makes space in the memory to store
    variables as model parameters. The
    following screenshot describes this process:


    ![](./images/c1b76ba2-079d-4dc2-a87f-a3af08ba6418.png)



2.  The model is built using the preceding hyperparameters. In our case,
    we have named the model `got2vec` ,but the model may be
    named as per your liking. The model definition is illustrated in the
    following screenshot:


    ![](./images/1655417a-5dec-4d80-8cc8-de25adb2f444.png)

3.  Running the `build_vocab` command on the model should
    produce an output as seen in the following screenshot:


    ![](./images/a259395a-b313-433f-9d24-ca89006a04dc.png)

4.  Training the model is done by defining the parameters as seen in the
    following screenshot:


    ![](./images/1326f92c-e13d-470c-886e-454de9cdc5ec.png)



5.  The above command produces an output as shown in the following
    screenshot:


![](./images/8433fc2c-dbd2-48cc-9dcc-faf9af5f3d2d.png)


6.  The commands to save, checkpoint, and load the model produce the
    following output, as shown in the screenshot:


![](./images/cd46369a-af17-4833-8d88-f492c821e494.png)


### There\'s more\...

Consider the following:


-   In our case, we notice the `build_vocab` function
    identifies 23,960 different word types from a list of 1,110,288
    words. However, this number will vary for different text corpora.
-   Each word is represented by a 300-dimensional vector since we have
    declared the dimensionality to be 300. Increasing this number
    increases the training time of the model
    but also makes sure the model generalizes easily to new data.
-   The downsampling rate of 1e


    ![](./images/a9a19a2d-830b-4235-ad91-afdda58f3d3e.png)


    3 is found to be a good rate. This is specified to let the
    model know when to downsample frequently
    occurring words, as they are not of much importance when it comes to
    analysis. Examples of such words are this, that, those, them, and so
    on.
-   A seed is set to make results reproducible. Setting a seed also
    makes debugging a lot easier.
-   Training the model takes about 30 seconds using regular CPU
    computing since the model is not very complex.
-   The model, when check-pointed, is saved under the
    `trained` folder inside the working directory.


### See also

For more information on `Word2Vec` models and the Gensim
library, visit the following link:

<https://radimrehurek.com/gensim/models/word2vec.html>

Visualizing further 
--------------------------------------

This section will describe how to squash the
dimensionality of all the trained words and put it all into one giant
matrix for visualization purposes. Since each word is a 300-dimensional
vector, it needs to be brought down to a lower dimension for us to
visualize it in a 2D space.


### Getting ready

Once the model is saved and checkpointed after training, begin by
loading it into memory, as you did in the previous section. The
libraries and modules that will be utilized in this section are: 


-   `tSNE`
-   `pandas`
-   `Seaborn`
-   `numpy`


### How to do it\...

The steps are as follows:

1.  Squash the dimensionality of the 300-dimensional word vectors by
    using the following command:

```
 tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
```


2.  Put all the word vectors into one giant matrix (named
    `all_word_vectors_matrix`), and view it using the
    following commands:

```
 all_word_vectors_matrix = got2vec.wv.syn0
 print (all_word_vectors_matrix)
```


3.  Use the `tsne` technique to fit all the learned
    representations into a two- dimensional space using the following
    command:

```
 all_word_vectors_matrix_2d =  tsne.fit_transform(all_word_vectors_matrix)
```


4.  Gather all the word vectors, as well as their associated words,
    using the following code:

```
 points = pd.DataFrame(
     [
            (word, coords[0], coords[1])
             for word, coords in [
              (word, all_word_vectors_matrix_2d[got2vec.vocab[word].index])
                for word in got2vec.vocab
         ]
    ],
    columns=["word", "x", "y"]
)
```


5.  The `X` and `Y` coordinates and associated words
    of the first ten points can be obtained using the following command:

```
points.head(10)
```


6.  Plot all the points using the following commands:

```
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(15, 15))
```


7.  A selected region of the plotted graph can be zoomed into for a
    closer inspection. Do this by slicing the original data using the
    following function:

```
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
        ]
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
        for i, point in slice.iterrows():
            ax.text(point.x + 0.005, point.y + 0.005, point.word,                                                  fontsize=11)
```


8.  Plot the sliced data using the following command. The sliced data
    can be visualized as a zoomed-in region of the original plot of all
    data points:

```
plot_region(x_bounds=(20.0, 25.0), y_bounds=(15.5, 20.0))
```


### How it works\...

The functionality is as follows:

1.  The t-SNE algorithm is a non-linear dimensionality reduction
    technique. Computers are easily able to interpret and process many
    dimensions during their computations. However, humans are only
    capable of visualizing two or three dimensions at a time. Therefore,
    these dimensionality reduction techniques come in very handy when
    trying to draw insights from data.


 

2.  On applying t-SNE to the 300-dimensional vectors, we are able to
    squash it into just two dimensions to plot it and view it.
3.  By specifying `n_components` as 2, we let the algorithm
    know that it has to squash the data into a two-dimensional space.
    Once this is done, we add all the squashed vectors into one giant
    matrix named `all_word_vectors_matrix`, which is
    illustrated in the following screenshot:


    ![](./images/f3c7b488-a037-45d9-86ca-486f85643a32.png)

4.  The t-SNE algorithm needs to be trained on all these word vectors.
    The training takes about five minutes on a regular CPU.
5.  Once the t-SNE is finished training on all the word vectors, it
    outputs 2D vectors for each word. These vectors may be plotted as
    points by converting all of them into a data frame. This is done as
    shown in the following screenshot:


    ![](./images/54711d7f-11ae-4e1f-ac77-19c8435fcc36.png)



6.  We see that the preceding code produces a number of points where
    each point represents a **`word`** along with its **`X`** and
    **`Y`** coordinates. On inspection of the first twenty points of the
    data frame, we see an output as illustrated in the following
    screenshot:


![](./images/b9ccd002-8bcb-44ae-a130-f91e2e44fd96.png)


7.  On plotting all the points using the
    `all_word_vectors_2D` variable, you should see an output
    that looks similar to the one in the following screenshot:


    ![](./images/6cd8ac65-f7a7-4775-ac3f-275e51c0ab9e.png)

8.  The above command will produce a plot of all tokens or words
    generated from the entire text as shown in the following screenshot:


    ![](./images/1fdd6ed9-060c-4021-885c-c4b51fe73b34.png)



9.  We can use the `plot_region` function to zoom into a
    certain area of the plot so that we are able to actually see the
    words, along with their coordinates. This step is illustrated in the
    following screenshot:


    ![](./images/79c069a0-d9c4-4112-b1b0-641abe6bebe0.png)



10. An enlarged or zoomed in area of the plot can be visualized by
    setting the `x_bounds` and `y_bounds`, values as
    shown in the following screenshot:


    ![](./images/3e07cb80-ca2a-4bd9-b777-23ca092deca3.png)



11. A different region of the same plot can be visualized by varying the
    `x_bounds` and `y_bounds` values as shown in the
    following two screenshots:


    ![](./images/77a17732-2b12-48ad-b638-0d77edfb9621.png)



![](./images/b3efd6b0-527f-48e3-8804-7acbeb07800b.png)


### See also

The following additional points are of note:


-   For more information on how the t-SNE algorithm works, visit the
    following link:
-   <https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm>
-   More information about cosine distance similarity and ranking can be
    found by visiting the following
    link:<https://code.google.com/archive/p/word2vec/>
-   Use the following link to explore the
    different functions of the `Seaborn`
    library:<https://seaborn.pydata.org/>


Analyzing further
-----------------------------------

This section will describe further analysis that can be performed on the
data after visualization. For example,
exploring cosine distance similarity between different word vectors.


### Getting ready

The following link is a great blog on how cosine distance similarity
works and also discusses some of the math involved:

<http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/>

### How to do it\...

Consider the following:


-   Various natural-language processing tasks can be performed using the
    different functions of `Word2Vec`. One of them is finding
    the most semantically similar words given a certain word (that is,
    word vectors that have a high cosine similarity or a short Euclidean
    distance between them). This can be done by using
    the `most_similar` function form `Word2Vec`, as
    shown in the following screenshot:


    ![](./images/23ee51a5-b9de-4af2-be6d-02ed85fbdafb.png)


    This screenshots  all the closest words related to the word
    `Lannister`:


    ![](./images/3b2321b3-98c1-4c37-8b5a-4a713670afdc.png)


    This screenshot shows a list of all the words related to
    word `Jon`:


    ![](./images/3c32c59f-49ee-4089-a71c-e2fa3f7f0157.png)



### How it works\...

Consider the following:


-   There are various methods to measure the
    semantic similarity between words. The one we are using in this
    section is based on cosine similarity. We can also explore linear
    relationships between words by using the following lines of code:

```
 def nearest_similarity_cosmul(start1, end1, end2):
    similarities = got2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
)
start2 = similarities[0][0]
print("{start1} is related to {end1}, as {start2} is related to         {end2}".format(**locals()))
return start2
```



-   To find the cosine similarity of nearest words to a given set of
    words, use the following commands:

```
nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "dragons")
```



-   The preceding process is illustrated in the following screenshot:


    ![](./images/a2482cac-fca4-4c1d-827a-f19b24d2cc2a.png)

-   The results are as follows:


    ![](./images/0cbc97cc-16bd-40ac-b1d5-936859951214.png)

-   As seen in this section, word vectors form the basis of all NLP
    tasks. It is important to understand them and the math that goes
    into building these models before diving
    into more complicated NLP models such
    as **recurrent neural networks** and **Long Short-Term
    Memory** (**LSTM**) cells.


### See also

Further reading can be undertaken for a better understanding of the use
of cosine distance similarity, clustering and other machine learning
techniques used in ranking word vectors. Provided below are a few links
to useful published papers on this topic:


-   <https://s3.amazonaws.com/academia.edu.documents/32952068/pg049_Similarity_Measures_for_Text_Document_Clustering.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1530163881&Signature=YG6YjvJb2z0JjmfHzaYujA2ioIo%3D&response-content-disposition=inline%3B%20filename%3DSimilarity_Measures_for_Text_Document_Cl.pdf>
-   <http://csis.pace.edu/ctappert/dps/d861-12/session4-p2.pdf>
