def new(a,n):
	name='output_'+str(n)
	s=open(name,'w')
	s.write(a)
	s.close()
ss=open(r'/home/juicy/python/exp.txt','r')
char=ss.read(1)
i=1
while char:
	new(char,i)
	i=i+1
	char=ss.read(1)
ss.close();

