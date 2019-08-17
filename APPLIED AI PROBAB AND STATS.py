
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
     
printmd("bold text") # for bold text


# In[3]:


# Q-Q PLOT
import numpy as np
import pylab
import scipy.stats as stats

#N(0,1)
#generating obsns
# loc =mean, scale =stdd devn,size says generate samples 
#with mean m and scale 1 and 
#generate 1000 such samples
std_normal=np.random.normal(loc=0,scale=1,size=1000)

#0 to 100 th percentiles of std normal
for i in range(0,101):
    print(i,np.percentile(std_normal,i))


# In[4]:


#generate 100 samples from N(20,5)
measurements=np.random.normal(loc=20,scale=5,size=100)

# q-q plot comparsion very easy with pylab

# just compare measurements and a normal distribution
# analyse the generated line
stats.probplot(measurements,dist='norm',plot=pylab)
pylab.show()


# In[5]:


# all points almost lie on line hence gaussian
#but not all 
# is it incorrrect 
#no it is due to less sample size 
#increase sample size and see


measurements=np.random.normal(loc=20,scale=5,size=50000)
stats.probplot(measurements,dist='norm',plot=pylab)
pylab.show()


# In[6]:


#now we get perfect line

#same can be done for other types like uniform,normal etc

measurements=np.random.uniform(low=-1,high=1,size=5000)
stats.probplot(measurements,dist='norm',plot=pylab)
pylab.show()

#as this is unifrom and nt normal


# In[7]:


#code 2

import random
print(random.random()) #any no. bw 0 and 1
#everytime we get different values
 


# In[8]:


#say we wanna sample uniformly
# doesn work but method is this only
import pandas as pd
iris=datasets.load_iris()
d=iris.data
d.shape


#sample 30 points randoly froM 150 point datset
n=150;
m=30;
p=m/n;

sampled_data=[];

for i in range(0,n):
    if random.random() <=p:
        sampled_data.append(iris[i,:])
len(sampled_data)

#note the size of sample willbe roughly 30 not perfect 30



# In[ ]:


#BOX COX TRANSFORM FOR PARETO DISTRIBUTION 
#TO CONVERT TO GAUSSIAN

from scipy import stats
import matplotlib.pyplot as plt

#generating a non pareto distribution
#checking q-q plot
fig=plt.figure()
ax1=fig.add_subplot(211)
x=stats.loggamma.rvs(5,size=5000) +5 #pareto distbtn for 500 points
prob=stats.probplot(x,dist=stats.norm,plot=ax1) #q-q plot
ax1.set_xlabel('')
ax1.set_title('probplot against normal distribution')


#now use box cox 
ax2=fig.add_subplot(212)
xt,_=stats.boxcox(x) #take values into xt 
prob=stats.probplot(xt,dist=stats.norm,plot=ax2)
#now after box cox we can see that it becomes gaussian
ax2.set_title('probplot after box cox transform')
plt.show()


# In[ ]:


#bootstrap samples
import numpy as np
from pandas import read_csv
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

#loading datset
x=np.array([180,162,158,172,168,150,171,183,165,176])

#configure bootstrap
n_iterations=1000  #k
n_size=int(len(x))

#run bootstrap
medians=list()
for i in range(n_iterations):
    #preapre train and test series
    s=resample(x,n_samples=n_size); #for simplicity let m =k
    m=np.median(s);
    #print(m)
    medians.append(m)
    
#plot scores
pyplot.hist(medians)
pyplot.show()

#confidence intervals

alpha=0.95
p=((1.0-alpha)/2.0)*100
lower=np.percentile(medians,p)

p=(alpha+((1.0-alpha)/2.0))*100
upper=np.percentile(medians,p)

print('c.i 95' ,(alpha*100,lower,upper))
    


# In[ ]:


#note ci wide coz less size of original data


# In[ ]:


#k-s test
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

#generate a gaussian r.v. X
x=stats.norm.rvs(size=1000); #normal distbn
sns.set_style('whitegrid')
sns.kdeplot(np.array(x),bw=0.5)
plt.show() 


# In[ ]:


#now  for ks test just one line
stats.kstest(x,'norm')

