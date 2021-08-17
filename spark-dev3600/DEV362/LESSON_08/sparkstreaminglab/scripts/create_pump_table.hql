CREATE EXTERNAL TABLE pump_info

(resourceid STRING, type STRING, purchasedate STRING,
dateinservice STRING, vendor STRING, longitude FLOAT, latitude FLOAT)

ROW FORMAT DELIMITED FIELDS TERMINATED BY ","


STORED AS TEXTFILE LOCATION "/headless/Desktop/next-level-python-big-data/spark-dev3600/sensorvendor.csv";
