from underthesea import word_tokenize
import re

def split_word(list):
    list = [word_tokenize(line) for line in list]
    return list

def remove_stopword(list):
    stopWord =[]
    for line in open('StopWords.txt', 'r' , encoding="utf8"):
        stopWord.append(line.strip())
    data = []
    dline = []
    for line in list:
        for word in line:
            if word in  stopWord:
                continue
            dline.append(word)
        data.append(dline)
        dline = []

    return data

def remove_number(list):
    output = [re.sub(r'\d+', ' ', line.lower()) for line in list]
    return output

def remove_punctuation(list):
    list = [re.sub(r'[^\w\s]',' ',line.lower()) for line in list]
    return list


def remove_space(list):
    list = [" ".join(line.split()) for line in list]
    return list

def check_vietnamese(list, content):
    syllables = content.split()
    number_vietnamese_syllables = 0
    for syllable in syllables:
        if syllable not in list:
            continue
        number_vietnamese_syllables += 1
    return (number_vietnamese_syllables / len(syllables)) > 0.5

def remove_not_vietnamese(list):
    vietnamese_dictionary = []
    for line in open('syllables_dictionary_1.txt', 'r', encoding="utf8"):
        vietnamese_dictionary.append(line.strip())
    data_list = []
    dline = []
    for line in list:
        content = ' '.join(line[1:])
        if not check_vietnamese(vietnamese_dictionary, content):
            continue
        for word in line:
            dline.append(word)
        data_list.append(dline)
        dline = []
    return data_list

if __name__ == '__main__':
    train = []
    for line in open('sentiment_analysis_train.v1.0.txt', 'r' , encoding="utf8"):
        train.append(line.strip())

    # train = remove_punctuation(train)
    # train = remove_number(train)
    # train = remove_space(train)
    train = split_word(train)
    # train = remove_stopword(train)
    # train = remove_not_vietnamese(train)
    count = 0
    f = open("trainSV.txt", "w", encoding="utf8")

    for line in train:
        for word in line:
            count +=1
        if (count > 1):
            for word in line:
                f.write("%s " % word)
            f.write("\n")
        count = 0
    f.close()