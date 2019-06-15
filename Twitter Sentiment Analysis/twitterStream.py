from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt

conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
sc = SparkContext(conf=conf)

def main():
    #conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    #sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    
    sc.setLogLevel("WARN")
    
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)



def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    # YOUR CODE HERE
    
    pCounts = []
    nCounts = []
    
    for count in counts:
        for word in count:
            if word[0] == "positive":
                pCounts.append(word[1])
            else:
                nCounts.append(word[1])
    
    maxi = max(max(pCounts),max(nCounts))+110
    
    plt.axis([-1, len(pCounts), 0, maxi])
    
    pos, = plt.plot(pCounts, 'b-', marker = 'o', markersize = 5)
    neg, = plt.plot(nCounts, 'g-', marker = 'o', markersize = 5)
    plt.legend((pos,neg),('Positive','Negative'),loc=2)
    
    plt.xticks(np.arange(0, 13, 1))
    
    plt.xlabel("Time Step")
    plt.ylabel("Word Count")
    
    plt.savefig("plot.png")
    plt.show()



def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    #return [word for line in open(filename, 'r') for word in line.split()]
    
    rdd = sc.textFile(filename)
    return set(rdd.collect())
    

    
def classifyWords(word,pwords,nwords):
    if word in pwords:
        return "positive"
    elif word in nwords:
        return "negative"
    else:
        return None

    
    
def updateFunction(newValues, runningCount):
    if runningCount is None:
        runningCount = 0
    return sum(newValues, runningCount)
    
    
    
def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1])

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    
    words = tweets.flatMap(lambda line: line.split(" "))
    pairs = words.map(lambda word: (classifyWords(word,pwords,nwords), 1)).filter(lambda x: x[0]=="positive" or x[0] == "negative")

    wordCounts = pairs.reduceByKey(lambda x, y: x + y)
    
    running_counts = pairs.updateStateByKey(updateFunction)

    running_counts.pprint()
    
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    
    wordCounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    # YOURDSTREAMOBJECT.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)
    
    return counts



if __name__=="__main__":
    main()
