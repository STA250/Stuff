#!/bin/bash

HADOOP_HOME=/usr/local/Cellar/hadoop/1.2.1
JAR=libexec/contrib/streaming/hadoop-streaming-1.2.1.jar
HSTREAMING="$HADOOP_HOME/bin/hadoop jar $HADOOP_HOME/$JAR"

$HSTREAMING \
    -file mapper.py    -mapper mapper.py \
    -file reducer.py   -reducer reducer.py \
    -input test/pg* -output test-output
