import train
import Body
from Stemmer import Stemmer
import codecs
import nltk
import re
import math
import random
from nltk.corpus import stopwords
stop_words = stopwords.words('farsi')
stop_words.append(".")
stop_words.append(":")
stop_words.append("،")
import operator
import glob
stemmer = Stemmer()


def sigmoid(x):
  return (1 / (1 + math.exp(-x)))


def words_list(paragraph, stemmed_sent):
    ''' takes a text as an input and reduce it to embbeded list of stemmed-words of sentences (without stop words)'''
    fp = codecs.open(paragraph, 'r', 'utf8')
    txt = fp.read()
    fp.close()
    sentences = txt.split('\r\n')
    #print('sentences', sentences)
    embbeded = []
    for line in sentences:
        #print(line)
        if line != "\r":
            words = re.split('\u200c| |، |\r', line)
            embed_sent = []
            for word in words:
                if word != '' and word != '-' and word != '(' and word != ')' and word != '.':
                    if word not in stop_words:
                        stemmed_sent.append(stemmer.stem(word))
                    embed_sent.append(word)
            if len(embed_sent) != 0:
                embbeded.append(embed_sent) 
    return embbeded



### reading final weights from text file ###
fp1 = open('new_weights.txt', 'r')
txt1 = fp1.readlines()
i_2_h_weights = [[] for i in range(Body.p)]
h_2_o_weights = []
i = 0	
for line in txt1:
    #print(line)
    if line == '\n':
        break
    i_2_h_weights[i] = line.split()     #loading weight matrix for hidden layer
    i += 1

h_2_o_weights = txt1[-3].split()	    #loading weight matrix for output layer
bais1 = txt1[-2].split()

### change strings to integers for weight values  ###
bias2 = float(txt1[-1])
for i in range(Body.p):
    for j in range(Body.n):
        i_2_h_weights[i][j] = float(i_2_h_weights[i][j])
for i in range(Body.p):
    h_2_o_weights[i] = float(h_2_o_weights[i])
    bais1[i] = float(bais1[i])
fp1.close()


path = 'C:/Users/sara/Desktop/Summerize_Corpus/test_tags/*.txt' 
files = glob.glob(path)
overall_tags = []
for file in files:
    fp2 = open(file)
    txt2 = fp2.read()
    fp2.close()
    tags = txt2.split('\n')[1:-1]
    for i in tags:
        overall_tags.append(int(i))


path = 'C:/Users/sara/Desktop/Summerize_Corpus/test_corpus/*.txt' 
files = glob.glob(path)
doc_vectors = []
file_number = 0


output1 = []
output = []
for file in files:
    ''' feature extraction '''
    para_vector = []
    stemmed_sent = []
    para_sents = words_list(file, stemmed_sent)
    fd = nltk.FreqDist(stemmed_sent)   #finding thematic words
    thematic_words = sorted(fd.items(), key=operator.itemgetter(1))
    ten_thematic_words = thematic_words[-10:]
    thematics = []
    for i in range(len(ten_thematic_words)):
        thematics.append(ten_thematic_words[-(i+1)])
        dict_thematics = dict(thematics)
    #print(thematics)
    title_words = []                   #finding the title words
    for w in para_sents[0]:
        if w not in stop_words:
            title_words.append(stemmer.stem(w))
    len_sents = []
    for sentence in para_sents[1:]:     #compute the length of sentences to find the maximum sentence length in a paragraph then
        len_sents.append(len(sentence))
    
    sentense_number = 0                  #finding sentence location
    for sentence in para_sents[1:]:
        if para_sents.index(sentence) == 1: #checks if it's the first sentence
            sent_vector = [1, 1 / (len(para_sents)-1)]
        else:
            sent_vector = [0]
            sent_vector.append(para_sents.index(sentence) / (len(para_sents)-1))
        sent_vector.append(len(sentence) / max(len_sents))   #computes the relative length of  sentence
        
        count1 = 0                     # counts the number of thematic words in a sentence
        count2 = 0                     # counts the number of title words in a sentence
        flag_numerical = 0
        for word in sentence:        
            if stemmer.stem(word) in dict_thematics.keys():
                count1 += 1
                #print(count)
            if stemmer.stem(word) in title_words:
                count2 += 1
            if word.isdigit():   # see if any numerical data exists in a sentence
                flag_numerical = 1
        sent_vector.append(count1 / len(sentence))
        sent_vector.append(count2 / len(title_words))
        if flag_numerical == 1:
            sent_vector.append(1)
        else:
            sent_vector.append(0)
        #print(sent_vector)
        #para_vector.append(sent_vector)



            
        ####      training        ####  
        ''' feed_forward '''
        input_x = sent_vector
        z_ins = []
        for j in range(len(bais1)):
            z_in = bais1[j]
            for i in range(Body.n):
                z_in = z_in + (i_2_h_weights[j][i] * input_x[i])
                z = sigmoid(z_in)
            #zed_ins.append(z_in)
            z_ins.append(z)
        y_in = bias2
        for k in range(len(z_ins)):
           y_in = y_in + (z_ins[k] * h_2_o_weights[k])
           y = (sigmoid(y_in)) * (10 ** 4)
        output1.append(y - int(y))
        if y - int(y) > 0.5:
            output.append(1)
        else:
            output.append(0)
    file_number += 1
print('correct_tags','\t',  'extracted_tags')
for i in range(len(overall_tags)):
    print('\t', overall_tags[i], '\t\t', output[i])
### precision & recall ###
counter1 = 0
for i in range(len(overall_tags)):
    if output[i] == overall_tags[i]:
        counter1 += 1
print ("accuracy = %", float(counter1 / len(overall_tags)) * 100)
counter2 = 0
for j in output:
    if j == 1:
        counter2 += 1


counter3 = 0
counter4 = 0
for k in range(len(overall_tags)):
    if overall_tags[k] == 1:
        if output[k] == 1:
            counter3 += 1
        if output[k] != 1:
            counter4 += 1
       
print ("precision = %", float(counter3 / counter2) * 100)
print ("recall = %", float(counter3 / len(output)) * 100)

  

