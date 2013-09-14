<!--
<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
-->

### Coding Practice ###

Some basic coding problems to get you back in the swing.

1. Write a program that prints the numbers from 1 to 100.
But for multiples of three print "Fizz" instead of the
number and for the multiples of five print "Buzz".
For numbers which are multiples of both three and 
five print "FizzBuzz". 
(From: <http://www.codinghorror.com/blog/2007/02/why-cant-programmers-program.html>)

2. Write a program that generates 1000 uniform random numbers
between 0 and ![equation](http://latex.codecogs.com/gif.latex?2%5Cpi) (call this ![equation](http://latex.codecogs.com/gif.latex?x)), 
and 1000 uniform random 
numbers between 0 and 1 (call this ![equation](http://latex.codecogs.com/gif.latex?y)).
You will then have 
10,000 pairs of random numbers.
Transform ![equation](http://latex.codecogs.com/gif.latex?%28x%2Cy%29) to ![equation](http://latex.codecogs.com/gif.latex?%28u%2Cv%29) where: 
![equation](http://latex.codecogs.com/gif.latex?u%3Dy*%5Ccos%28x%29%2C), and, 
![equation](http://latex.codecogs.com/gif.latex?v%3Dy*%5Csin%28x%29%2C)
<!--
$$ 
u = y * \cos(x) , \qquad
v = y * \sin(x)
$$
-->
Make a 2D scatterplot of the 10,000 (u,v) pairs.
What is the distribution of:  ![equation](http://latex.codecogs.com/gif.latex?r%3D%5Csqrt%28u%5E2%2Bv%5E2%29)?

3. Consider the following snippet: 

	"Hello, my name is Bob. I am a statistician. I like statistics very much."

  a. Write a program to spit out every character in the snippet 
to a separate file (i.e., file `out_01.txt` would contain the character `H`,
file `out_02.txt` would contain `e` etc.). Note that the `,` and spaces
should also get their own files. 

  b. Write a program to combine all files back together into a single file
that contains the original sentence. **Take care to respect whitespace
and punctuation!**



