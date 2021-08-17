CREATE EXTERNAL TABLE maint_table

(resourceid STRING, eventDate STRING,
technician STRING, description STRING)

ROW FORMAT DELIMITED FIELDS TERMINATED BY ","


STORED AS TEXTFILE LOCATION "/headless/Desktop/next-level-python-big-data/spark-dev3600/sensormaint.csv";
