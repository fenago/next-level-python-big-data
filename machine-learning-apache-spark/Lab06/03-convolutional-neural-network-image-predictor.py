#!/usr/bin/python

""" 03-convolutional-neural-network-image-predictor.py: Classify images without labels using a pre-trained Convolutional Neural Network """

# (1) Import the required PySpark and Spark Deep Learning libraries
from sparkdl import DeepImagePredictor
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema

# (2) Create a Spark Session using the Spark Context instantiated from spark-submit
spark = SparkSession.builder.appName("Convolutional Neural Networks - Deep Image Predictor").getOrCreate()

# (3) Load the assortment of images into a Spark DataFrame
assorted_images_df = ImageSchema.readImages("./data/image-recognition-data/assorted")

# (4) Predict what the objects in the non-labelled images are (top 10 classifications) using a pre-trained CNN
deep_image_predictor = DeepImagePredictor(inputCol="image", outputCol="predicted_label", modelName="InceptionV3", decodePredictions=True, topK=10)
predictions_df = deep_image_predictor.transform(assorted_images_df)
predictions_df.select("image.origin", "predicted_label").show(truncate=False)
