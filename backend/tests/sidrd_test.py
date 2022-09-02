"""Module to test report related functionalities."""
from datetime import datetime
from controllers import get_tokenized_reports
from models import TokenizedReport

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
        text2 = "I googled test_1.js and there was no way i know it is a test"
        text3 = "I found a bug in artifacts.py. I've created a PR in https://gitfoo.com/foo/bar/issues/1"
        text4 = "Small typo in artifacts.py:1:16. No need to PR now, but it should be fixed later on some small refactoring"

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

    # def test_retrain(self):
    #     tokenizer = Tokenizer()
    #     vectorizer = Vectorizer()

        # text1 = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        # text2 = "I googled test_1.js and there was no way i know it is a test"
        # text3 = "I found a bug in artifacts.py. I've created a PR in https://gitfoo.com/foo/bar/issues/1"
        # text4 = "Small typo in artifacts.py:1:16. No need to PR now, but it should be fixed later on some small refactoring"

    #     tokens1 = tokenizer.tokenize(text1, 'stem')
    #     tokens2 = tokenizer.tokenize(text2, 'stem')
    #     tokens3 = tokenizer.tokenize(text3, 'stem')
    #     tokens4 = tokenizer.tokenize(text4, 'stem')

    #     tokens = [tokens1, tokens2, tokens3, tokens4]
    #     features = vectorizer.retrain(tokens)

    #     self.assertEqual(features.shape, (4, len(vectorizer._Vectorizer__VECTORIZER.get_feature_names_out())))

class TestClusterizer(BaseTest):
    """TestCase class for Clusterizer funcionalities"""

    def test_clusterize(self):
        tokenizer = Tokenizer()
        vectorizer = Vectorizer()
        clusterizer = Clusterizer(vectorizer=vectorizer, limit=2)

        text1 = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        text2 = "I googled test_1.js and there was no way i know it is a test"
        text3 = "I found a bug in artifacts.py. I've created an issue in https://foozilla.com/foo/bar/issues/1. I do not want to fix it"
        text4 = "Small typo in artifacts.py:1:16. No need to pull request now, but it should be fixed later on some small refactoring"
        report = "Error found in artifacts.py. A new PR is opened in https://foozilla.com/foo/bar/pr/2. Think it is a small bug"

        r1 = TokenizedReport(
            report_id=1, creation_time=datetime.now(), status="RESOLVED",
            component="foo", dupe_of=None, summary=text1, comments=[],
            text=text1, tokens=tokenizer.tokenize(text1, 'stem')
        )
        r2 = TokenizedReport(
            report_id=2, creation_time=datetime.now(), status="RESOLVED",
            component="foo", dupe_of=1, summary=text2, comments=[],
            text=text2, tokens=tokenizer.tokenize(text2, 'stem')
        )
        r3 = TokenizedReport(
            report_id=3, creation_time=datetime.now(), status="RESOLVED",
            component="Artifacts", dupe_of=None, summary=text3, comments=[],
            text=text3, tokens=tokenizer.tokenize(text3, 'stem')
        )
        r4 = TokenizedReport(
            report_id=4, creation_time=datetime.now(), status="RESOLVED",
            component="Artifacts", dupe_of=None, summary=text4, comments=[],
            text=text4, tokens=tokenizer.tokenize(text4, 'stem')
        )
        rep = TokenizedReport(
            report_id=5, creation_time=datetime.now(), status="NEW",
            component="Artifacts", dupe_of=None, summary=report, comments=[],
            text=report, tokens=tokenizer.tokenize(report, 'stem')
        )

        test_db_reports = get_tokenized_reports(limit=50)
        test_db_reports.extend([r1,r2,r3,r4])

        clusters = clusterizer.clusterize(rep, test_db_reports)

        # Assert report_id is in clusters dataframe
        self.assertTrue(rep.report_id in clusters['report_id'].values)
        # Assert clusters is smaller than test_db_reports
        self.assertTrue(len(clusters) < len(test_db_reports))
        # Assert r3 is in clusters dataframe
        self.assertTrue(r3.report_id in clusters['report_id'].values)
