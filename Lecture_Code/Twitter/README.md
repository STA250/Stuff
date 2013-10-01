## Crude Twitter Stream Sentiment Map

> Pre-step: Follow the intstructions in Lecture 2 to fork and download the repo to your local machine. Make sure you have the latest version of the GitHub repo (using `git pull`) prior to running this code.

As explained in Lecture 01, the code in this directory can be executed to produce
a very crude Twitter sentiment analysis. 

Full information about the Twitter API is available at <https://dev.twitter.com/docs>

The basic steps are as follows:

+ Create a Twitter account at <http://twitter.com>. (You don't need to start tweeting ;)

+ Log in to the Twitter developer site with your new Twitter account <https://dev.twitter.com/apps>

+ Click on "Create a new application"

+ Fill out the form (just make up a website if you don't plan on using a real one)

+ Once created, click on "Create my access token"

+ Open `my_twitterstream.py` and copy in your twitter credentials (`access_token_key` and `access_token_secret`).

+ You are now ready to get Twitter-sentiment-analyzing! :) Basic steps are below:

To stream:

    python my_twitterstream.py > tweetdump.txt

or run `grab_stream.sh` by executing:

    ./grab_stream.sh

Note: you may need to first make `grab_stream.sh` executable by typing:

    chmod u+x grab_stream.sh

(Same thing applies to the scripts listed below). To analyze for "sentiment" by location (where available):

    ./make_locresults.sh

(Or: run `python tweet_location.py AFINN-111.txt tweetdump.txt > locresults.txt`) To plot the results:

    ./plot_map.sh

(Or run `R --no-save --vanilla < plot_map.R`). You should get an output file `tweetmap.pdf`.

Feel free to play around with all of these files and modify them to do new things. For debugging info you
can set `verbose` and `geo_verbose` to `True` in `tweet_location.py`.

**Note:** The geo-tagging used here is *very* limited. It only uses explicit geolocation data
which is only available for a small percentage of tweets (exercise: modify the code to report
what fraction of tweets have geolocation information). Location data can also be gleaned from
many other attributes of the user profile, but this is omitted here for simplicity.





