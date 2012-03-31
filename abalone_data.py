# Module for processing the abalone dataset
import pickle
from numpy import *
import re
import random as modrandom
import os.path

DATA_PATH = ".\\data\\"

def read_data_file():
    afile = open(DATA_PATH+"abalone.data",'r')  # open the file
    lines = afile.readlines()
    afile.close()

    # Now randomly permute the lines, to remove any particular order
    modrandom.shuffle(lines)
    
    # Preallocate space
    xarr = zeros([4177,10],dtype=float) 
    yarr = zeros([4177,1],dtype=int)
    linecount=0
    for line in lines:
        elems = re.split(",",line)
        y = elems.pop()
        y = int(y.rstrip())
        sex = elems[0]
        sex_vect = [0,0,0]
        if sex is 'M':
            sex_vect[0]=1
        elif sex is 'F':
            sex_vect[1]=1
        else:
            sex_vect[2]=1
        remainder = [float(elem) for elem in elems[1:]]
        x = sex_vect + remainder

        # Now set the values in the array:
        yarr[linecount] = y
        xarr[linecount] = x
        linecount+=1
    return (yarr,xarr)

def cook_data(raw):
    y = raw[0]
    x = raw[1]
    # Subtract the mean from x
    mean = x.mean(axis=0)
    x_adj = x-mean
    # Rescale for unit variance
    std = x_adj.std(axis=0)
    x_final = x_adj/std
    return (y,x_final)

def pickle_data(data,name):
    output = open(DATA_PATH+name+".dat",'w')
    pickle.dump(data,output)
    output.close()

def data_exists():
    return os.path.isfile(DATA_PATH+"raw.dat") and os.path.isfile(DATA_PATH+"cooked.dat")

def unpickle_data():
    raw = open(DATA_PATH+"raw.dat",'r')
    cooked = open(DATA_PATH+"cooked.dat",'r')
    data = (pickle.load(raw),pickle.load(cooked))
    raw.close()
    cooked.close()
    return data

def get_data():
    if data_exists():
        return unpickle_data()
    else:
        raw = read_data_file()
        pickle_data(raw,"raw")
        cooked = cook_data(raw)
        pickle_data(cooked,"cooked")
        return (raw,cooked)
