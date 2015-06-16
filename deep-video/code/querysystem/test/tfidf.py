from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
if __name__ == '__main__':
    train = ['the game of life is a everlasting learning', 'the unexamined life is not worth living', 'never stop learning']
    test = ['the sun in the sky is bright', 'we can see the shining sun, the bright sun'] 
   
    tf = TfidfVectorizer()
    x = tf.fit_transform(train)
    print x
    print tf.get_feature_names()
    idf = tf.idf_
    print dict(zip(tf.get_feature_names(), idf))
    cv = CountVectorizer()
    cv.fit_transform(train)
    print cv.vocabulary
    freq_term_matrix = cv.transform(test)
    print freq_term_matrix.todense()
   # tf = TfidfVectorizer(analyzer = 'word', min_df = 0, stop_words = 'english', strip_accents = 'unicode')
   # tfidf_matrix = tf.fit_transform(corpus)
   # feature_names = tf.get_feature_names()
