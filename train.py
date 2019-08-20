
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
    sentences = txt.split("\n")
    #print(sentences)
    embbeded = []
    for line in sentences:
        if line != "\r":
            words = re.split('\u200c| |، |\r', line)
            embed_sent = []
            for word in words:
                if word != ' ' and word != '-' and word != '(' and word != ')' and word != '.':
                    if word not in stop_words:
                        stemmed_sent.append(stemmer.stem(word))
                    embed_sent.append(word)
            embbeded.append(embed_sent)
    return embbeded



alpha = 0.5  #نرخ يادگيري

### reading initial weights from text file ###
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


path = 'C:/Users/sara/Desktop/ai_projet2/Summerize_Corpus/train_corpus/*.txt' 
files = glob.glob(path)
doc_vectors = []
file_number = 0
#input_x = []
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
        input_x = (int(Body.overall_tags[file_number][sentense_number]), sent_vector)
        sentense_number += 1
        zed_ins = [] 
        z_ins = []

        

        
        for j in range(len(bais1)):
            z_in = bais1[j]
            for i in range(Body.n):
                z_in = z_in + (i_2_h_weights[j][i] * input_x[1][i])
                z = sigmoid(z_in)
            zed_ins.append(z_in)
            z_ins.append(z)
        y_in = bias2
        for k in range(len(z_ins)):
           y_in = y_in + (z_ins[k] * h_2_o_weights[k])
           y = sigmoid(y_in)
           
        ''' back_propagation'''
        sigma = (input_x[0] - y)*(y * (1 - y))
        #print('correct tag ', input_x[0])
        #print('sigma ', sigma)
        if sigma < 0:   ### threshold for output layer
          new_ws = []
          all_new_vs = []
          first_biases = []
          for i in range(Body.p):
            delta_w = sigma * alpha * z_ins[i]
            sigma_j = (sigma * h_2_o_weights[i]) * (zed_ins[i]*(1 - zed_ins[i]))
            new_w = h_2_o_weights[i] + delta_w           ### update second layer weights
            new_ws.append(new_w)
            new_vs = []
            for j in range(Body.n):
              delta_v = sigma_j * alpha * input_x[1][j]
              new_v = i_2_h_weights[i][j] * delta_v    ### update first layer weights
              new_vs.append(new_v)
            all_new_vs.append(new_vs)
            delta_bias1 = sigma_j * alpha    ### update first layer biases
            first_biases.append(delta_bias1)
          delta_bias2 = sigma * alpha          ### update second layer biases
          fp2 = open('new_weights.txt' , 'w')  ### write new weights in a text file for next round
          for i in all_new_vs:
            for j in i:
              fp2.write(str(j) + '  ')
            fp2.write('\n')
          fp2.write('\n')
          for w in new_ws:
            fp2.write(str(w) + '  ')
          fp2.write('\n')
          for i in first_biases:
            fp2.write(str(i) + '  ')
          fp2.write('\n')
          fp2.write(str(delta_bias2))
          fp2.close()
    
    file_number += 1


  

