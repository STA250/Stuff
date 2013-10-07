<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    },
  TeX: { equationNumbers: { autoNumber: "AMS" } }    
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

STA 250 :: Advanced Statistical Computation
================================


*For all questions you must show your work. This enables us to understand your thought process, give partial credit and prevent crude cheating. Please see the code of the conduct in the Syllabus for rules about collaborating on homeworks.*

*For questions requiring computing, if you use `R`, `python` or any programming environment then you must turn in a printout of your output with your solutions.
In addition, a copy of your code must be uploaded to your `HW0` directory as per Q6 below.*

<br/>

## Homework 0 (No Credit -- Practice Only) ##
## Due: In Class, 5:30pm Wed October 9th ##

#### Assigned: Wednesday Oct 2nd

Some basic coding problems to get you back in the swing.

1. Write a program that prints the numbers from 1 to 100.
But for multiples of three print "Fizz" instead of the
number and for the multiples of five print "Buzz".
For numbers which are multiples of both three and 
five print "FizzBuzz". 
(From: <http://www.codinghorror.com/blog/2007/02/why-cant-programmers-program.html>)

2. Write a program that generates 10,000 uniform random numbers
between 0 and ![equation](http://latex.codecogs.com/gif.latex?2%5Cpi) (call this ![equation](http://latex.codecogs.com/gif.latex?x)), 
and 10,000 uniform random 
numbers between 0 and 1 (call this ![equation](http://latex.codecogs.com/gif.latex?y)).
You will then have 
10,000 pairs of random numbers.<br/>
Transform ![equation](http://latex.codecogs.com/gif.latex?%28x%2Cy%29) to ![equation](http://latex.codecogs.com/gif.latex?%28u%2Cv%29) where: 
![equation](http://latex.codecogs.com/gif.latex?u%3Dy*%5Ccos%28x%29%2C), and, 
![equation](http://latex.codecogs.com/gif.latex?v%3Dy*%5Csin%28x%29).<br/>
Make a 2D scatterplot of the 10,000 (u,v) pairs.
What is the distribution of:  ![equation](http://latex.codecogs.com/gif.latex?r%3D%5Csqrt%28u%5E2%2Bv%5E2%29)?

3. Consider the following snippet:

		Hello, my name is Bob. I am a statistician. I like statistics very much.

	a. Write a program to spit out every character in the snippet 
to a separate file (i.e., file `out_01.txt` would contain the character `H`,
file `out_02.txt` would contain `e` etc.). Note that the `,`, `.` and spaces
should also get their own files. 

	b. Write a program to combine all files back together into a single file
that contains the original sentence. **Take care to respect whitespace
and punctuation!**

4. Run `boot_camp_demo.py` as a batch job on `Gauss` using the submission script `boot_camp_sarray.sh` in the Github repo. Follow the instructions in class for how to do this. Note: You will want to fork and clone the course repo if you have not done so already, please see Q7 for details.

5. Run the Twitter code provided in lecture. Make sure to run the tweet-grabbing portion of code for a sufficient length of time (It is recommended to open another terminal and run `ls -alh` to check the size of the output file). The `README` provides full instructions for each of the steps.
  + See how your plot differs from the one shown in lecture 01
  + Modify the code to report the percentage of tweets that had geo-tagged data at the end of the sentiment analysis.

6. Consider the autoregressive process of order 1, usually called an AR(1) process:<br/>

	![equation](http://latex.codecogs.com/gif.latex?y_t%3D%5Crho_t%20y_%7Bt-1%7D%20%2B%20%5Cepsilon_t)
	<br/>
	for t=1,2,…,n. Let ![equation](http://latex.codecogs.com/gif.latex?y_0%3D0) and ![equation](http://latex.codecogs.com/gif.latex?%5Cepsilon_t%20%5Csim%20N%280%2C1%29) being independent. 
	
	a. Simulate from this process with n=1000. Plot the resulting series.
	b. Repeat part (a) 200 times, storing the result in a 1000x200 matrix. Each column should correspond to a realization of the random process.
	c. Compute the mean of the 200 realizations at each time point (i=1,2,…,1000). Plot the means.
	d. Plot the variance of the 200 realizations at each time point (i=1,2,…,1000). Plot the variances. 
	e. Compute the mean of each of the 200 series across time points (j=1,2,…,200). Plot the means.
	f. Compute the variance of each of the 200 series across time points (j=1,2,…). Plot the variances.
	g. Justify the results you have seen in parts b.--f. theoretically.
	
7. 
	a. Let ![equation](http://latex.codecogs.com/gif.latex?Z%5Csim%7B%7DN%280%2C1%29). 
Compute ![equation](http://latex.codecogs.com/gif.latex?E%5B%5Cexp%5E%7B-Z%5E%7B2%7D%7D%5D) using Monte Carlo integration.<br/>
	b. Let ![equation](http://latex.codecogs.com/gif.latex?Z%5Csim%7B%7DTruncated-Normal%280%2C1%3B%5B-2%2C1%5D%29). 
Compute ![equation](http://latex.codecogs.com/gif.latex?E%5BZ%5D) using importance sampling.

8. Let ![equation](http://latex.codecogs.com/gif.latex?x_%7Bij%7D%5Csim%7B%7DN%280%2C1%29) for i=1,...,n and j=1,2, ![equation](http://latex.codecogs.com/gif.latex?x_%7Bi0%7D%3D1)
for i=1,...,n, ![equation](http://latex.codecogs.com/gif.latex?x_%7Bi%7D%5E%7BT%7D%3D%28x_%7Bi0%7D%2Cx_%7Bi1%7D%2Cx_%7Bi2%7D%29%5E%7BT%7D),
![equation](http://latex.codecogs.com/gif.latex?%5Cbeta%3D%281.2%2C0.3%2C-0.9%29%5E%7BT%7D) and 
![equation](http://latex.codecogs.com/gif.latex?%5Cepsilon_%7Bi%7D%5Csim%7B%7DN%280%2C1%29) for i=1,...,n.
Simulate from the linear regression model with n=100. Use the bootstrap procedure to estimate the
SE of ![equation](http://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta%7D) based on B=1000 bootstrap resamples.

7. In this question you will fork the course GitHub repo and upload your homework code to Q1-Q5 to the repo. Go to <https://github.com/STA250/Stuff>.

  + Click on the "Fork" button: <br/>
<img src="https://raw.github.com/exosamsi/detrending/master/fork.png">
  + Wait for the fork to complete. When it does, you will be taken to the newly forked repo. For example:<br/>
<img src="Screenshot 2013-09-26 10.54.38.png"><br/>
This has forked the repo to your GitHub.com account, but the repo is not stored on your laptop/desktop at this point.
On Mac/Windows, load the GitHub GUI you should have installed, and then proceed as below:
  + Go to the GitHub GUI, click on your username under the "GitHub.com" tab, click on "Refresh" (on the bottom of the screen). The forked repo should appear with the option to "Clone to Computer". For example:<br/>
<img src="Screenshot 2013-09-26 10.56.22.png" style="width: 600px;">
  + Click "Clone to Computer". Select where to save the repo, and wait for the clone to complete:<br/>
<img src="Screenshot 2013-09-26 10.58.21.png" style="width: 600px;"><br/>
Congratulations, you have now successfully forked the course repo. Any time you need to update the repo (e.g., if Prof. Baines posts new code/slides/assignments) you can click on "My Repositories" and the forked "Stuff" repo. Then click on the "Sync Branch" icon in the top left of the GUI:<br/>
<img src="Screenshot 2013-09-26 11.08.38.png">
<br/>
Note: In general, you will need to commit any changes for the sync to proceed. 

  + For the Linux-ers on Mac command-line folks, you can do all of the above via (with obvious modifications):

		git config --global user.name "John Doe"
		git config --global user.email johndoe@example.com
		git clone https://github.com/[yourgithubusername]/[yourforkedrepo].git

	To get the current status (note: you must be in the local repo directory):
	
		cd Stuff # Change to the newly downloaded repo
		ls # Check files are there… 
		git status # Should be up-to-date
	
	To get the latest updates:

		git pull
	
	Again, any local changes must be committed prior to the `git pull` request.

  + Now, move your `HW0.(R|py)` into the `HW0` directory on your local machine. If you now run `git status` (or check the status via the GUI) you should receive a message informing you that you have uncommitted files. 
  + If using the command line, run (from the local repo directory):
		
		git add HW0/HW0.py
	
	or similar, depending on what your code file is called. This will add the file to the repo.
  + Next, commit the change to the repo (run this from the local repo directory):
	
		git commit -m
		
	Add a message for the update e.g., "Added HW0 code". This stages the commit on your local machine, but the commit will not appear on GitHub.com until you push it to the site. If using the GUI you can commit via the "Changes" tab. In the "Commit Summary" box, enter a message for the update, then click "Commit". The committed changes should appear in the "Unsynced Commits" tab. 		

  + To push the change to GitHub, from the local repo directory:
	
		git push 
			
	Using the GUI, push to GitHub using the "Sync Branch" button. 


  + Voila. If you go to your GitHub account, and navigate to the `HW0` folder of the forked repo, you should see your homework code! :)
	

### (: Happy Coding! :) ###


