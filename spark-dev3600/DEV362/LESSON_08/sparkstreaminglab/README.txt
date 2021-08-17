Create an hbase table to write to:
$ hbase shell
> create '/headless/Desktop/next-level-python-big-data/spark-dev3600/sensor', {NAME=>'data'}, {NAME=>'alert'}, {NAME=>'stats'}

Commands to run labs:

Step 1: First compile the project on eclipse: Select project -> Run As -> Maven Install
Step 2: scp sparkstreaminglab-1.0.jar user01@ipaddress:/headless/Desktop/next-level-python-big-data/spark-dev3600/.
Step 3: spark-submit --class solutions.SensorStream --master local[2] sparkstreaminglab-1.0.jar
Step 4: cp sensordata.csv  /headless/Desktop/next-level-python-big-data/spark-dev3600/stream/.
Step 5: Run other examples:
            spark-submit --class solutions.SensorStreamSQL --master local[2] sparkstreaminglab-1.0.jar
            spark-submit --class solutions.SensorStreamWindow --master local[2] sparkstreaminglab-1.0.jar
            spark-submit --class solutions.HBaseSensorStream --master local[2] --driver-memory 256m sparkstreaminglab-1.0.jar
