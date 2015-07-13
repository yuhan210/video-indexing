from nltk.corpus import stopwords

word_dd = {}

for line in open('8k-pictures.html').readlines():
    line = line.strip()
    if line.find('li') >= 0 and line.find('td') < 0:
        words = line.split('<li>')[-1].split(' ')
        for w in words:
            w = w.lower()
            if w in stopwords.words('english'):
                continue
            if w not in word_dd:
                word_dd[w] = 1
            else:
                word_dd[w] += 1

print sorted(word_dd.items(), key=lambda x: x[1], reverse=True)[:100]
