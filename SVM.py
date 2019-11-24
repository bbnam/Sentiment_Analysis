from underthesea import word_tokenize
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer
)
label_map = {
        '__label__trung_binh': 0,
        '__label__kem': 1,
        '__label__rat_kem': 2,
        '__label__tot': 3,
        '__label__xuat_sac': 4
    }
data_with_label = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}
def calculateTfidf(
        data, count_vectorizer=None, tf_idf_transformer=None
):
    if not (count_vectorizer and tf_idf_transformer):
        count_vectorizer = CountVectorizer()
        tf_idf_transformer = TfidfTransformer()
        x_counts = count_vectorizer.fit_transform(data)
        x_tf_idf = tf_idf_transformer.fit_transform(x_counts)
    else:
        x_counts = count_vectorizer.transform(data)
        x_tf_idf = tf_idf_transformer.transform(x_counts)
    return [count_vectorizer, tf_idf_transformer, x_counts, x_tf_idf]

def open_data_from_file():
    with open('trainSVM.txt') as file:
        for line in file:
            if line[-1] == '\n':
                data = line[:-1]
            else:
                data = line
            split_data = data.split(' ')
            label = split_data[0]
            content = ' '.join(split_data[1:])
            if label not in label_map:
                continue
            data_with_label[label_map[label]].append(content)
    return data_with_label


import pickle
def train_SVM():

    x_train = []
    y_train = []
    data_with_label = open_data_from_file()
    clf = LinearSVC(C=0.1)

    for label, contents in data_with_label.items():
        for content in contents:
            content = content.lower()
            words = word_tokenize(content)
            new_words = list(map(
                lambda word: '_'.join(word.split(' ')), words)
            )
            content_after_handling = ' '.join(new_words)
            x_train.append(content_after_handling)
            y_train.append(label)
    (count_vectorizer, tf_idf_transformer, x_train_counts,
     x_train_tf_idf) = calculateTfidf(x_train)
    clf.fit(x_train_tf_idf, y_train)

    f = open("train.pkl", "wb")
    pickle.dump([count_vectorizer, tf_idf_transformer, x_train_counts, x_train_tf_idf, clf], f)
    f.close()

def predict_SVM(file):
    x_test = []

    (count_vectorizer,tf_idf_transformer, x_train_counts,
     x_train_tf_idf, clf) = pickle.load(open("train.pkl", "rb"))

    with open(file) as file:
        for line in file:
            x_test.append(line)

    (count_vectorizer, tf_idf_transformer, x_test_counts,
     x_test_tf_idf) = calculateTfidf(
        x_test,
        count_vectorizer,
        tf_idf_transformer
    )

    max_label_map = []

    for data_test in x_test_tf_idf:
        y_predict = clf.predict(data_test.toarray())
        max_label_map.append(y_predict[0])

    max_keys_list =[]
    for index in max_label_map:
        for key in label_map.keys():
            if label_map[key] == index:
                max_keys_list.append(key)
    return max_keys_list

if __name__ == '__main__':

    # train_SVM()
    max_keys_list = predict_SVM('sentiment_analysis_test.v1.0.txt')
    f = open("kqSVM.txt", "w", encoding="utf8")

    for key in max_keys_list:
        f.write("%s " % key)
        f.write("\n")
    f.close()


