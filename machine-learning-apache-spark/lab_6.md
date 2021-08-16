
Deep Learning Using Apache Spark
================================

In this lab, we will go on a hands-on exploration on Multilayer perceptrons

Case study 1 -- OCR
===================

A great real-world use case to demonstrate the power of MLPs is that of
OCR. In OCR, the challenge is to recognize human writing, classifying
each handwritten symbol as a letter. In the case of the English
alphabet, there are 26 letters. Therefore, when applied to the English
language, OCR is actually a classification problem that has *k* = 26
possible classes!

**Note:**

The dataset that we will be using has been derived from the **University
of California\'s** (**UCI**) Machine Learning Repository, which is found
at <https://archive.ics.uci.edu/ml/index.php>. The specific letter
recognition dataset that we will use, available from both the GitHub
repository accompanying this course and from
<https://archive.ics.uci.edu/ml/datasets/letter+recognition>, was
created by David J. Slate at Odesta Corporation; 1890 Maple Ave; Suite
115; Evanston, IL 60201, and was used in the paper *Letter Recognition
Using Holland-style Adaptive Classifiers* by P. W. Frey and D. J. Slate
(from Machine Learning Vol 6 \#2 March 91).



MLPs in Apache Spark
====================

Let\'s return to our dataset and train an MLP in Apache Spark to
recognize and classify letters from the English alphabet. If you open
[ocr-data/letter-recognition.data] in any text editor, from either
the GitHub repository accompanying this course or from UCI\'s machine
learning repository, you will find 20,000 rows of data, described by the
following schema:

![](./images/3.png)

This dataset describes 16 numerical attributes representing statistical
features of the pixel distribution based on scanned character images,
such as those illustrated in *Figure 7.5*. These attributes have been
standardized and scaled linearly to a range of integer values from 0 to 15. 
For each row, a label column called [lettr] denotes the letter
of the English alphabet that it represents, where no feature vector maps
to more than one class---that is, each feature vector maps to only one
letter in the English alphabet.



Let\'s now use this dataset to train an MLP classifier to recognize
symbols and classify them as letters from the English alphabet:

**Note:**

The following subsections describe each of the pertinent cells in the
corresponding Jupyter notebook for this use case, called
[01-multilayer-perceptron-classifier.ipynb]. This notebook
can be found in the GitHub repository accompanying this course.


1.  First, we import the prerequisite PySpark libraries as normal,
    including [MLlib]\'s [MultilayerPerceptronClassifier]
    classifier and [MulticlassClassificationEvaluator] evaluator
    respectively, as shown in the following code:

```
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
```


2.  After instantiating a Spark context, we are now ready to ingest our
    dataset into a Spark dataframe. Note that in our case, we have
    preprocessed the dataset into CSV format, where we have converted
    the [lettr] column from a [string] datatype to a
    [numeric] datatype representing one of the 26 characters in
    the English alphabet. This preprocessed CSV file is available in the
    GitHub repository accompanying this course. Once we have ingested this
    CSV file into a Spark dataframe, we then generate feature vectors
    using [VectorAssembler], comprising the 16 feature columns, as
    usual. The resulting Spark dataframe, called [vectorised\_df],
    therefore contains two columns---the numeric [label] column,
    representing one of the 26 characters in the English alphabet, and
    the [features] column, containing our feature vectors:

```
letter_recognition_df = sqlContext.read
.format('com.databricks.spark.csv')
.options(header = 'true', inferschema = 'true')
.load('letter-recognition.csv')
feature_columns = ['x-box','y-box','width','high','onpix','x-bar',
'y-bar','x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy',
'y-ege','yegvx']
vector_assembler = VectorAssembler(inputCols = feature_columns,
outputCol = 'features')
vectorised_df = vector_assembler.transform(letter_recognition_df)
.withColumnRenamed('lettr', 'label').select('label', 'features')
```


3.  Next, we split our dataset into training and test datasets with a
    ratio of 75% to 25% respectively, using the following code:

```
train_df, test_df = vectorised_df
.randomSplit([0.75, 0.25], seed=12345)
```


4.  We are now ready to train our MLP classifier. First, we must define
    the size of the respective layers of our neural network. We do this
    by defining a Python list with the following elements:
    
    -   The first element defines the size of the input layer. In our
        case, we have 16 features in our dataset, and so we set this
        element to [16].
    -   The next elements define the sizes of the intermediate hidden
        layers. We shall define two hidden layers of sizes [8] and
        [4] respectively.
    -   The final element defines the size of the output layer. In our
        case, we have 26 possible classes representing the 26 letters of
        the English alphabet, and so we set this element to [26]:

```
layers = [16, 8, 4, 26]
```


5.  Now that we have defined the architecture of our neural network, we
    can train an MLP using [MLlib]\'s
    [MultilayerPerceptronClassifier] classifier and fit it to the
    training dataset, as shown in the following code. Remember that
    [MLlib]\'s [MultilayerPerceptronClassifier] classifier
    uses the sigmoid activation function for hidden neurons and the
    softmax activation function for output neurons:

```
multilayer_perceptron_classifier = MultilayerPerceptronClassifier(
maxIter = 100, layers = layers, blockSize = 128, seed = 1234)
multilayer_perceptron_classifier_model =
multilayer_perceptron_classifier.fit(train_df)
```


6.  We can now apply our trained MLP classifier to the test dataset in
    order to predict which of the 26 letters of the English alphabet the
    16 numerical pixel-related features represent, as follows:

```
test_predictions_df = multilayer_perceptron_classifier_model
.transform(test_df)
print("TEST DATASET PREDICTIONS AGAINST ACTUAL LABEL: ")
test_predictions_df.select("label", "features", "probability",
"prediction").show()

```


7.  Next, we compute the accuracy of our trained MLP classifier on the
    test dataset using the following code. In our case, it performs very
    poorly, with an accuracy rate of only 34%. We can conclude from this
    that an MLP with two hidden layers of sizes 8 and 4 respectively
    performs very poorly in recognizing and classifying letters from
    scanned images in the case of our dataset:

```
prediction_and_labels = test_predictions_df.select("prediction", "label")
accuracy_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
precision_evaluator = MulticlassClassificationEvaluator(metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
print("Accuracy on Test Dataset = %g" % accuracy_evaluator.evaluate(prediction_and_labels))
print("Precision on Test Dataset = %g" % precision_evaluator.evaluate(prediction_and_labels))
print("Recall on Test Dataset = %g" % recall_evaluator.evaluate(prediction_and_labels))


Accuracy on Test Dataset = 0.339641
Precision on Test Dataset = 0.313333
Recall on Test Dataset = 0.339641
```


8.  How can we increase the accuracy of our neural classifier? To answer
    this question, we must revisit our definition of what the hidden
    layers do. Remember that the job of the neurons in the hidden layers
    is to learn to detect patterns within the input data. Therefore,
    defining more hidden neurons in our neural architecture should
    increase the ability of our neural network to detect more patterns
    at greater resolutions. To test this hypothesis, we shall increase
    the number of neurons in our two hidden layers to 16 and 12
    respectively, as shown in the following code. Then, we retrain our
    MLP classifier and reapply it to the test dataset. This results in a
    far better performing model, with an accuracy rate of 72%:

```
# (9) To improve the accuracy of our model, let us increase the size of the Hidden Layers
new_layers = [16, 16, 12, 26]
new_multilayer_perceptron_classifier = MultilayerPerceptronClassifier(maxIter=400, layers=new_layers, blockSize=128, seed=1234)
new_multilayer_perceptron_classifier_model = new_multilayer_perceptron_classifier.fit(train_df)
new_test_predictions_df = new_multilayer_perceptron_classifier_model.transform(test_df)
print("New Accuracy on Test Dataset = %g" % accuracy_evaluator.evaluate(new_test_predictions_df.select("prediction", "label")))
```


