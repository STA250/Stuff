import sys
import json

def hw(sent_file,tweet_file,verbose=False,geo_verbose=False):

    sentiments = sent_file.readlines()
    tweets = tweet_file.readlines()
    sent_file_nlines = len(sentiments)
    tweet_file_nlines = len(tweets)
    print "sentiment,lat,lon"
    if verbose:
        print "Lines in sentiment file: " + str(sent_file_nlines)
        print "Lines in tweet file:     " + str(tweet_file_nlines)
    ################################################
    # Parse sentiment file:
    sents, scores = [],[]
    for row in sentiments:
        rs = row.split("\t")
        tmp_word = rs[0].decode('utf-8')
        tmp_score = rs[1].replace("\n","")
        if verbose:
            print tmp_word + ": " + tmp_score
        sents.append(tmp_word)
        scores.append(tmp_score)
    if verbose:
        print "Finished parsing sentiment file..."
    ################################################
    tscores = []
    line_ctr = 0
    tweet_key = u'text'
    for line in tweets:
        json_tweet = json.loads(line)
        #if verbose:
        #    print "Keys:"
        #    print json_tweet.keys()
        if tweet_key in json_tweet.keys():
            ########################################
            # Sentiment stuff...
            ########################################
            tweet_exists = True
            tweet = json_tweet[tweet_key]
            tweet_words = tweet.split()
            tmp_score = 0.0
            if verbose:
                print "Tweet exists!"
                print "Actual tweet: " + tweet
                print 'Computing sentiment for terms in tweet ' + str(line_ctr)
            for word in tweet_words:
                lc_word = word.lower()
                if verbose:
                    print lc_word
                # Find sentiment:
                if lc_word in sents:
                    word_score = scores[sents.index(lc_word)]
                    if verbose:
                        print "Matched!"
                        print "Sentiment of " + word + " = " + str(word_score)
                    tmp_score = tmp_score + float(word_score)
                else:
                    if verbose:
                        print "Not matched"
            # Finish checking all words in tweet:
            tscores.append(tmp_score)
            if verbose:
                print "Sentiment score for tweet " + str(line_ctr) + ":"

            ########################################
            # Location stuff...
            ########################################
            geo_key = u'geo'
            type_key = u'type'
            point_key = u'Point'
            coord_key = u'coordinates'
            if geo_key in json_tweet.keys():
                geo_stuff = json_tweet[geo_key]
                if geo_stuff is not None:
                    if geo_verbose:
                        print "Geo stuff: " + str(geo_stuff)
                    #{u'type': u'Point', u'coordinates': [29.64896515, -82.35765414]}
                    if type_key in geo_stuff:
                        if geo_stuff[type_key] == point_key:
                            if verbose:
                                print "Geo stuff has co-ordinates..."
                            print str(tmp_score) + "," + str(geo_stuff[coord_key][0]) + "," + str(geo_stuff[coord_key][1])
                else:
                    if verbose:
                        print "No geo stuff :("
            else:
                if verbose:
                    print "Tweet has no geo key! :("
            # END location stuff..
        else:
            tweet_exists = False
            if verbose:
                print "No tweet here!"
        line_ctr = line_ctr+1
    if verbose:
        print "Finished computing (new) sentiments for all tweets! :)"

def lines(fp):
    print str(len(fp.readlines()))

def main():
    verbose=False
    geo_verbose=False
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    hw(sent_file=sent_file,tweet_file=tweet_file,verbose=verbose,geo_verbose=geo_verbose)
    #lines(sent_file)
    #lines(tweet_file)

if __name__ == '__main__':
    main()
