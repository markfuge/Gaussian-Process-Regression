

# Import the Necessary Modules
# GPRegression module
from gpregressor import *
# Kernal Module?
# Abalone Processing module
import abalone_data
# Plotting modules
import matplotlib
from pylab import plot, show, ylim, yticks
import matplotlib.pyplot as plt


def graph_abalone_data(y,x,title,fignum=None):
    att_names = ["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]
    if fignum is None:
        plt.figure()
    else:
        plt.figure(fignum)
    for i in range(0,len(att_names)):
        plt.subplot(4,2,i)
        plt.scatter(x[:,i+3],y, faceted=False)
        plt.xlabel(att_names[i])
    #plt.title(title)
    plt.show()

# Load and process the dataset
raw,cooked = abalone_data.get_data()

# plot the data, to get some idea about how it looks
y = cooked[0]
x = cooked[1]
#graph_abalone_data(y,x,"Cooked",1)
#graph_abalone_data(raw[0],raw[1],"Raw",2)

num_partions = 5
# Initialize the regressor
gpr = GPRegressor(x,y,num_partions)

# Test Values
sigma = [0.01, 0.03, 0.1, 0.3, 1., 3.]
Lambda = gpr.return_quantiles()
results=np.empty((len(sigma),len(Lambda),2))

# Run Tests
output = open('results.csv','w')

# Keep the same lambda across different sigmas, to save kernal computation
for v,l in enumerate(Lambda):
    gpr.reset_lambda(l)
    for u,s in enumerate(sigma):
        avg_v_error = 0
        avg_t_error = 0
        for i in range(0,num_partions):
            v_err,t_err = gpr.estimate_mean(i,s,l)
            avg_v_error += v_err
            avg_t_error += t_err
        avg_v_error /= num_partions
        avg_t_error /= num_partions
        outstring = 's: {}, l: {}, verr: {}, terr: {}\n'.format(s,l,avg_v_error,avg_t_error)
        print outstring
        output.write(outstring)

output.close()
