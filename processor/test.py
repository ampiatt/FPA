import gensim
import tensorflow as tf
import csv

def clean(items):
    for item in items:
        if item == '&amp;':
            continue
        if item.startswith('b\'') or item.startswith('b"'):
            item = item[2:]
        if item.startswith('\'') or item.startswith('"'):
            item = item[1:]
        if item.endswith('\'') or item.endswith('"'):
            item = item[:-1]
        yield item

def sentences():
    with open('Combined_News_DJIA.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            i = 0
            for title in row:
                if i < 2:
                    i += 1
                    continue
                yield [item for item in clean(title.split())]

data = []
for sentence in sentences():
    data.append(sentence)
data = data[25:]
model = gensim.models.Word2Vec(data, min_count=1, workers=4, seed=1234)

print(model.wv.similarity("Russia", "NATO"))
print(model.wv.similarity("DJIA", "NATO"))
print(model.wv.similarity("DJIA", "Dogs"))

for word in data[0]:
    print('{}: {}'.format(word, model.wv.similarity('DJIA', word)))
