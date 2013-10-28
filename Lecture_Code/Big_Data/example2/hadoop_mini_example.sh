#!/bin/bash

HADOOP_HOME=/usr/local/Cellar/hadoop/1.2.1
JAR=libexec/contrib/streaming/hadoop-streaming-1.2.1.jar
HSTREAMING="$HADOOP_HOME/bin/hadoop jar $HADOOP_HOME/$JAR"

# Ideally, groups would be spread across many files
# so that map jobs are more efficiently distributed

$HSTREAMING \
    -file mapper.py    -mapper mapper.py \
    -file reducer.py   -reducer reducer.py \
    -input data/mini_groups* -output mini-groups-output
