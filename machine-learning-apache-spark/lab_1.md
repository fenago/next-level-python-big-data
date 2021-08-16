Developing Spark Application
=============================

In this lab, we will write our first spark application.


You can access the Python shell and execute interactive Python commands as follows:

```
> python

>>> import sys
>>> sys.path
>>> quit()
```

#### Spark Application
We are  ready to write our first Spark application in Python!
Instantiate a Jupyter Notebook instance, access it via your internet
browser, and create a new Python 3 notebook containing the following
code (it may be easier to split the following code over separate
notebook cells for future ease of reference):

```
# (1) Import required Python dependencies
from pyspark import SparkContext, SparkConf
import random

# (2) Instantiate the Spark Context
conf = SparkConf()
.setMaster("local")
.setAppName("Calculate Pi")
sc = SparkContext(conf=conf)

# (3) Calculate the value of Pi i.e. 3.14...
def inside(p):
x, y = random.random(), random.random()
return x*x + y*y < 1

num_samples = 100
count = sc.parallelize(range(0, num_samples)).filter(inside).count()
pi = 4 * count / num_samples

# (4) Print the value of Pi
print(pi)

# (5) Stop the Spark Context
sc.stop()
```


This PySpark application, at a high level, works as follows:

1.  Import the required Python dependencies, `including pyspark`
2.  Create a Spark context, which tells the Spark application how to
    connect to the Spark cluster, by instantiating it with a
    [SparkConf] object that provides application-level settings at
    a higher level of precedence
3.  Calculate the mathematical value of Pi π
4.  Print the value of Pi and display it in Jupyter Notebook as a cell
    output
5.  Stop the Spark context that terminates the Spark application

If you access the Spark Master web UI before executing
[sc.stop()], the Spark application will be listed under [Running
Applications], at which time you may view its underlying
worker and executor log files. If you access the Spark Master web UI
following execution of [sc.stop()], the Spark application will be
listed under [Completed Applications].

**Note:**

Note that this notebook can be downloaded from the GitHub repository
accompanying this course and is called `01-test-jupyter-notebook-with-pyspark.ipynb`.
