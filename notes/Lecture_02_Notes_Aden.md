Things you Should've done already:
==================================
* Set up a github
* Set up an account 

Fun Exercise:
============
For numbers i=1, …, 100:

* If the number is divisible by 3, print "Fizz"
* If divisible by 5, print "Buzz"
* If divisible by both, print "FizzBuzz"
* Otherwise, print the number itself

SSH:
====
Secure Shell.
Syntax: ssh username@server
Ex: ssh pdbaines@gauss.ucdavis.edu

For X11 forwarding (for graphics and windows), SSH -X pdbaines@gauss.ucdavis.edu

Can be annoying to keep typing the full ssh command. Try writing a shell script or something like that to make it shorter.

> echo "ssh -Xv pdbaines@gauss.cse.ucdavis.edu" > ssh_logon.sh

> chmod 755 ssh_logon.sh

Once on Gauss, you have the usual array of Unix commands.

Github:
=======
Version control system that allows you get copies of files posted on the course site, edit them, share them with yourself and others…
Usually done from command line, but a Github GUI can make things a little easier if you aren't used to the terminal syntax.

> git config --global user.name "First Last"

> git config --global user.email "email@ucdavis.edu"

These two commands set up your contact information on the machine, on your commit messages.
To set up a git repository, find the address for the git repo you want, then type:

###A couple useful Git commands:###
* git clone: Set up a repository from an existing Git repo
* git add: Add files to a commit stage
* git commit: Commit your code locally to be submitted
* git push: Submit the committed code to Github
* git pull: Download the most recent files from Github.
* git status: Get the status on your current repo

A note on writing software on the server:
=========================================
You will not have the same tools you use on your local machine (likely your laptop). It may be better to write code on your laptop, then push the code to Gauss using SCP/SFTP.

Gauss:
======
Gauss is not an interactive system--you cannot run your session interactively and still utilize the cluster structure.

Batch File:
-----------
* SBATCH: Name of the job
* SARRAY: Number of times to run the job (run a job 20 times: you should do range=0, 1-19)

To actually run the file:
> stun python file_name.py -o out_${SLURM_ARRAY_ID}.txt

This creates a series of files called {out_1.txt, out_2.txt, …}.

Python:
=======
If Windows: Download from Python website

For Mac and Linux: You already have it installed! To make sure you are running the correct version:
 > python --version

There are currently two major supported versions: Python 2 and Python 3--they do not work with each other!

R:
==
Download from the R webpage.  
Tutorials:

* <http://www.ats.ucla.edu/stat/r/>
* <http://www.codeschool.com/courses/try-r>

Good IDEs:
==========
Python: Spyder, PyCharm, Eclipse, IDLE  
R: RStudio