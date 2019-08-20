import random
import math
import glob
#import feature_vector

### initial weights with nogen_vidro method
n = 6
p = 5
beta = 0.7 * (p ** (1/n))

i_2_h_weights = [[random.uniform(-0.5, 0.5) for i in range(n)] for j in range(p)]
temp = 0
for raw in range(len(i_2_h_weights)):
    for weight in i_2_h_weights[raw]:
        temp = temp + (weight ** 2)
    i2h_length = temp ** (1/2)
    for w in range(n):
        i_2_h_weights[raw][w] = (beta * i_2_h_weights[raw][w]) / i2h_length

#bias1 = random.uniform(-beta, beta)
h_2_o_weights = [random.uniform(-0.5, 0.5) for i in range(p)]
temp = 0
for weight in h_2_o_weights:
    temp = temp + (weight ** 2)
h2o_length = temp ** (1/2)
for w in range(p):
    h_2_o_weights[w] = (beta * h_2_o_weights[w]) / h2o_length               

first_layer_biases = []
for j in range(p):
    bias1 = random.uniform(-beta, beta)  #generates bias for hidden layer
    first_layer_biases.append(bias1)

bias2 = random.uniform(-beta, beta)     #generates bias for output layer

### write initial weights in a file ###
fp2 = open('new_weights.txt' , 'w')
for i in i_2_h_weights:
    for j in i:
        fp2.write(str(j) + ' ')
    fp2.write('\n')
fp2.write('\n')
for w in h_2_o_weights:
    fp2.write(str(w) + ' ')
fp2.write('\n')
for bias in first_layer_biases:
    fp2.write(str(bias) + ' ')
fp2.write('\n')
fp2.write(str(bias))
fp2.close()


path = 'C:/Users/sara/Desktop/ai_project2/Summerize_Corpus/train_tags/*.txt' 
files = glob.glob(path)
overall_tags = []
for file in files:
    fp = open(file)
    txt = fp.read()
    fp.close()
    tags = txt.split('\n')[1:-1]
    #print(tags)
    overall_tags.append(tags)
