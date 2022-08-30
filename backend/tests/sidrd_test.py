"""Module to test report related functionalities."""
from datetime import datetime

from sidrd import SIDRD, Tokenizer, Vectorizer, Clusterizer, Classifier
from tests import BaseTest

from sklearn.feature_extraction.text import TfidfVectorizer

class TestSIDRD(BaseTest):
    """TestCase class for SIDRD duplicate detection funcionalities"""

    pass

class TestTokenizer(BaseTest):
    """TestCase class for Tokenizer funcionalities"""

    def test_tokenize(self):
        text = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(text, 'lemmatize')
        self.assertEqual(tokens,  ['test', 'also', 'know', 'test1', 'test_1', 'js', 'google', 'google'])
        tokens = tokenizer.tokenize(text, 'stem')
        self.assertEqual(tokens, ['test', 'also', 'know', 'test1', 'test_1', 'js', 'googl', 'googl'])

class TestVectorizer(BaseTest):
    """TestCase class for Vectorizer funcionalities"""

    def test_vectorize(self):
        tokenizer = Tokenizer()
        vectorizer = Vectorizer()

        text1 = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        text2 = "I can't believe it's working!"
        text3 = "I found a bug in artifacts.py. I've created a PR in https://gitfoo.com/foo/bar/issues/1"
        text4 = "Error found in artifacts.py. I've created a PR in https://gitfoo.com/foo/bar/issues/2. Think it is a small bug"

        tokens1 = tokenizer.tokenize(text1, 'stem')
        tokens2 = tokenizer.tokenize(text2, 'stem')
        tokens3 = tokenizer.tokenize(text3, 'stem')
        tokens4 = tokenizer.tokenize(text4, 'stem')

        shape = len(vectorizer._Vectorizer__VECTORIZER.get_feature_names_out())

        # Vectorize one text
        features1 = vectorizer.vectorize([tokens1])
        self.assertEqual(features1.shape, (1, shape))

        # Vectorize multiple texts
        features = vectorizer.vectorize([tokens1, tokens2, tokens3, tokens4])
        self.assertEqual(features.shape, (4, shape))

    def test_retrain(self):
        tokenizer = Tokenizer()
        vectorizer = Vectorizer()

        text1 = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        text2 = "I can't believe it's working!"
        text3 = "I found a bug in artifacts.py. I've created a PR in https://gitfoo.com/foo/bar/issues/1"
        text4 = "Error found in artifacts.py. I've created a PR in https://gitfoo.com/foo/bar/issues/2. Think it is a small bug"

        tokens1 = tokenizer.tokenize(text1, 'stem')
        tokens2 = tokenizer.tokenize(text2, 'stem')
        tokens3 = tokenizer.tokenize(text3, 'stem')
        tokens4 = tokenizer.tokenize(text4, 'stem')

        tokens = [tokens1, tokens2, tokens3, tokens4]
        features = vectorizer.retrain(tokens)

        self.assertEqual(features.shape, (4, len(vectorizer._Vectorizer__VECTORIZER.get_feature_names_out())))

# class TestClusterizer(BaseTest):
#     """TestCase class for Clusterizer funcionalities"""

#     def test_clusterize(self):
#         tokenizer = Tokenizer()
#         vectorizer = Vectorizer()
#         clusterizer = Clusterizer()

#         text1 = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
#         text2 = "I can't believe it's working!"
#         text3 = "I found a bug in artifacts.py. I've created a PR in https://gitfoo.com/foo/bar/issues/1"
#         report = "Error found in artifacts.py. I've created a PR in https://gitfoo.com/foo/bar/issues/2. Think it is a small bug"

#         tokens1 = tokenizer.tokenize(text1, 'stem')
#         tokens2 = tokenizer.tokenize(text2, 'stem')
#         tokens3 = tokenizer.tokenize(text3, 'stem')
#         tokens_report = tokenizer.tokenize(report, 'stem')

#         features = vectorizer.vectorize([tokens1, tokens2, tokens3, tokens_report])

#         clusters = clusterizer.clusterize(features, 2)
#         self.assertEqual(clusters.shape, (4,))
#         self.assertEqual(clusters[0], 0)
#         self.assertEqual(clusters[1], 0)
#         self.assertEqual(clusters[2], 1)
#         self.assertEqual(clusters[3], 1)