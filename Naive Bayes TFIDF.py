import math
from underthesea import word_tokenize

def split_word(list):
    list = [word_tokenize(line) for line in list]
    return list

def split_label(train):
    aLabel = []
    count = 0
    for line in train:
        for label in aLabel:
            if label == line[0]:
                count = count +1
        if count == 0: aLabel.append(line[0])
        count = 0
    return aLabel

def dict_label(train):
    label = dict()
    for i in split_label(train):
        label[i] = 0
    return label

def count_word(train):
    data = dict()
    for line in train:
        for word in line:
            if word == line[0]:
                continue
            key = word
            if word in data.keys():
                data[key][line[0]] +=1
            else:
                data.setdefault(word , dict())
                data[key] = dict_label(train)
                data[key][line[0]] += 1
    return data

def count_all_word_in_label(train):
    label = dict_label(train)
    for line in train:
        for i in line:
            if i == line[0]:
                continue
            label[line[0]] += 1
    return label

def write_to_file(dict, file):
    file = open(file, 'w', encoding="utf8")
    label = dict
    for key, vla in label.items():
        file.write('%s:%s \n' % (key, vla))
    file.close()

def write_prior_to_file(dict):
    file = open("prior.txt",'w',encoding="utf8")
    label = dict
    for key, vla in label.items():
        file.write('%s:%s \n' % (key, vla))
    file.close()

def open_prior_from_file():
    d = {}
    with open("prior.txt") as f:
        for line in f:
            (key, val) = line.split(':')
            d[key] = val.strip()
    return d

import pickle
def write_likeihood_to_file(dict):
    f = open("likeihood.pkl", "wb")
    pickle.dump(dict, f)
    f.close()

def open_likeihood_from_file():
    data  = pickle.load(open("likeihood.pkl", "rb"))
    return data

def compute_IDF(list):
    N_doc = len(list)
    count = 0
    icount = count_word(list)
    idf = {}
    for word in icount.keys():
        for line in list:
            if word in line: count += 1

        idf[word] = math.log10(N_doc/count)
        count = 0
    f = open("IDF.pkl", "wb")
    pickle.dump(idf, f)
    f.close()
    return idf

def openIDF():
    data = pickle.load(open("IDF.pkl", "rb"))
    return data

def train_NaiveBayes(train):
    documents = count_all_word_in_label(train)
    N_doc = 0
    for i in documents.keys():
        N_doc += documents[i]
    prior = {}
    for i in documents.keys():
        prior[i] = (documents[i] / N_doc)
    write_prior_to_file(prior)

    c = likeihood= count_word(train)

    for key in c.keys():
        for vkey in c[key].keys():
            a = (c[key][vkey] +1) / (documents[vkey] + N_doc)
            likeihood[key][vkey] = a

    write_likeihood_to_file(likeihood)
    compute_IDF(train)


def compute_TF(line):
    tf = {}
    N_line = len(line)
    wordDict = dict.fromkeys(line, 0)
    for word in line:
        wordDict[word] +=1
    for word in wordDict.keys():
        tf[word] = wordDict[word] / N_line
    return tf

def predict_NaiveBayes(file):
    test = []
    for line in open(file, 'r', encoding="utf8"):
        test.append(line.strip())

    test = split_word(test)
    idf = openIDF()
    prior = open_prior_from_file()
    likeihood = open_likeihood_from_file()
    vmax = {}
    max_keys_list = []

    for line in test:
        max_keys = line
        tf = compute_TF(line)
        for key in prior.keys():
            clabel = math.log10(float(prior[key]))
            for i in line:
                if (i not in likeihood.keys()):
                    continue
                tfidf = tf[i]*idf[i]
                clabel = clabel + tfidf*math.log10(float(likeihood[i][key]))
            vmax[key] =clabel
        max_value = max(vmax.values())

        for i in vmax.keys():
            if (max_value == vmax[i]):
                max_keys = i;
                break
        max_keys_list.append(max_keys)

    return max_keys_list

if __name__ == '__main__':
    # train = []
    # for line in open('trainNB.txt', 'r', encoding="utf8"):
    #     train.append(line.strip())
    # train = split_word(train)
    # train_NaiveBayes(train)

    max_keys_list = predict_NaiveBayes('sentiment_analysis_test.v1.0.txt')
    f = open("kqNB.txt", "w", encoding="utf8")
    for key in max_keys_list:
        f.write("%s " % key)
        f.write("\n")

    f.close()
