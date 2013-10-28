## Readme for Hadoop jobs

0. Find a machine/cluster running Hadoop. For those running a 64-bit version
of Linux, you can obtain a local install Hadoop using 
<http://tutorialforlinux.com/2013/10/25/how-to-install-hadoop-on-ubuntu-13-10-saucy-step-by-step-guide> 
(note: you don't need to use the Oracle Java SDK, the standard Java SDK bundled
with Ubuntu works just fine too). 
For those on Mac (Mountain Lion), you can obtain a local install by following:
<http://blog.tundramonkey.com/2013/02/24/setting-up-hadoop-on-osx-mountain-lion>.
Else, you can use Amazon and find a machine image with Hadoop preconfigured.

1. Fire up the Hadoop cluster. For those running a local Mac install: 

   /usr/local/Cellar/hadoop/1.2.1/libexec/bin/start-all.sh

For those on Ubuntu/Linux: 
    
    /usr/local/hadoop/bin/start-all.sh

2. Get the files onto the HDFS:

    hadoop fs -mkdir test
    hadoop fs -copyFromLocal pg*.txt test/
    hadoop fs -ls test/

3.  Run the MapReduce job. For those with a Mac install run:

    HADOOP_HOME=/usr/local/Cellar/hadoop/1.2.1
    JAR=libexec/contrib/streaming/hadoop-streaming-1.2.1.jar
    HSTREAMING="$HADOOP_HOME/bin/hadoop jar $HADOOP_HOME/$JAR"

    $HSTREAMING \
        -file mapper.py    -mapper mapper.py \
        -file reducer.py   -reducer reducer.py \
        -input test/pg* -output test-output

For Linux, modify `HADOOP_HOME` and `JAR` accordingly. For convenience
this has been placed in the `hadoop_example.sh` script.

4. Check the status of the job using <http://localhost:50060>.

5. When completed successfully, check the output files, and 
you probably want to copy back to the regular filesystem:

    hadoop fs -ls test-output
    hadoop fs -copyToLocal test-output ./

6. Cleanup:

    hadoop fs -rmr test
    hadoop fs -rmr test-output

Note: `hadoop fs -rm` deletes single files, for recursive delete use `-rmr`. 


