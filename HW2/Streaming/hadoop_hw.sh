#!/bin/bash

#hadoop fs -mkdir data
#hadoop distcp s3://sta250bucket/bsamples data/
#hadoop fs -ls data/bsamples
#hadoop fs -copyToLocal data/bsamples/out_mini.txt ./
#cat out_mini.txt | ./mapper.py | sort -k1,1 | ./reducer.py
#./hadoop_hw.sh
#hadoop fs -copyToLocal binned-output ./

LOCATION="AMAZON"

if [ "$LOCATION" = "AMAZON" ]; then
    echo "Running with Amazon settings..."
    JAR=contrib/streaming/hadoop-streaming.jar
else
    echo "Running with local settings..."
    HADOOP_HOME=/usr/local/Cellar/hadoop/1.2.1
    JAR=libexec/contrib/streaming/hadoop-streaming-1.2.1.jar
fi

HSTREAMING="$HADOOP_HOME/bin/hadoop jar $HADOOP_HOME/$JAR"

# Ideally, groups would be spread across many files
# so that map jobs are more efficiently distributed

$HSTREAMING \
    -file mapper.py    -mapper mapper.py \
    -file reducer.py   -reducer reducer.py \
    -input data/bsamples/out* -output binned-output


