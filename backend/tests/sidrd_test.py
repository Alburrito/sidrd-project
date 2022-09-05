"""Module to test report related functionalities."""
from datetime import datetime
from controllers import create_tokenized_report, get_tokenized_reports
from models import TokenizedReport

from sidrd import SIDRD, Tokenizer, Vectorizer, Clusterizer, Classifier
from tests import BaseTest

class TestTokenizer(BaseTest):
    """TestCase class for Tokenizer funcionalities"""

    def test_tokenize(self):
        sidrd = SIDRD()
        tokenizer = sidrd.tokenizer
        text = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        tokens = tokenizer.tokenize(text, 'lemmatize')
        self.assertEqual(tokens,  ['test', 'also', 'know', 'test1', 'test_1', 'js', 'google', 'google'])
        tokens = tokenizer.tokenize(text, 'stem')
        self.assertEqual(tokens, ['test', 'also', 'know', 'test1', 'test_1', 'js', 'googl', 'googl'])

class TestVectorizer(BaseTest):
    """TestCase class for Vectorizer funcionalities"""

    def test_vectorize(self):
        sidrd = SIDRD()
        tokenizer = sidrd.tokenizer
        vectorizer = sidrd.vectorizer

        text1 = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        text2 = "I googled test_1.js and there was no way i know it is a test"
        text3 = "New tab address bar freezes until mouse is moved"
        text4 = "Small typo in artifacts.py:1:16. No need to pull request now"
        
        tokens1 = tokenizer.tokenize(text1)
        tokens2 = tokenizer.tokenize(text2)
        tokens3 = tokenizer.tokenize(text3)
        tokens4 = tokenizer.tokenize(text4)

        shape = len(vectorizer._Vectorizer__VECTORIZER.get_feature_names_out())

        # Vectorize one text
        features1 = vectorizer.vectorize([tokens1])
        self.assertEqual(features1.shape, (1, shape))

        # Vectorize multiple texts
        features = vectorizer.vectorize([tokens1, tokens2, tokens3, tokens4])
        self.assertEqual(features.shape, (4, shape))

class TestClusterizer(BaseTest):
    """TestCase class for Clusterizer funcionalities"""

    def test_clusterize(self):
        sidrd = SIDRD()
        tokenizer = sidrd.tokenizer
        clusterizer = sidrd.clusterizer

        text1 = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        text2 = "I googled test_1.js and there was no way i know it is a test"
        text3 = "New tab address bar freezes until mouse is moved"
        text4 = "Small typo in artifacts.py:1:16. No need to pull request now"
        report = "Address bar freezes when making new tabs"

        r1 = TokenizedReport(
            report_id=1, creation_time=datetime.now(), status="RESOLVED",
            component="foo", dupe_of=None, summary=text1, comments=[],
            text=text1+" foo", tokens=tokenizer.tokenize(text1+" foo")
        )
        r2 = TokenizedReport(
            report_id=2, creation_time=datetime.now(), status="RESOLVED",
            component="foo", dupe_of=1, summary=text2, comments=[],
            text=text2+" foo", tokens=tokenizer.tokenize(text2+" foo")
        )
        r3 = TokenizedReport(
            report_id=3, creation_time=datetime.now(), status="RESOLVED",
            component="Address Bar", dupe_of=None, summary=text3, comments=[],
            text=text3+" Address Bar", tokens=tokenizer.tokenize(text3+" Address Bar")
        )
        r4 = TokenizedReport(
            report_id=4, creation_time=datetime.now(), status="RESOLVED",
            component="Artifacts", dupe_of=None, summary=text4, comments=[],
            text=text4+" Artifacts", tokens=tokenizer.tokenize(text4+" Artifacts")
        )
        for r in [r1, r2, r3, r4]:
            create_tokenized_report(r.report_id, r.creation_time, r.status, r.component, r.dupe_of, r.summary, r.comments, r.text, r.tokens)

        rep = TokenizedReport(
            report_id=5, creation_time=datetime.now(), status="NEW",
            component="Address Bar", dupe_of=None, summary=report, comments=[],
            text=report, tokens=tokenizer.tokenize(report+" Address Bar")
        )

        test_db_reports = get_tokenized_reports(limit=50)

        similar_reports = clusterizer.clusterize(rep, test_db_reports)

        # Assert report_id is not in similar_reports dataframe
        self.assertTrue(rep.report_id not in similar_reports['report_id'].values)
        # Assert similar_reports is smaller than test_db_reports
        self.assertTrue(len(similar_reports) < len(test_db_reports))
        # Assert r3 is in similar_reports dataframe
        self.assertTrue(r3.report_id in similar_reports['report_id'].values)


class TestClassifier(BaseTest):
    """TestCase class for Classifier funcionalities"""

    def test_get_possible_duplicates(self):
        sidrd = SIDRD()
        tokenizer = sidrd.tokenizer
        clusterizer = sidrd.clusterizer
        classifier = sidrd.classifier

        text1 = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        text2 = "I googled test_1.js and there was no way i know it is a test"
        text3 = "New tab address bar freezes until mouse is moved"
        text4 = "Small typo in artifacts.py:1:16. No need to pull request now"
        report = "Address bar freezes when making new tabs"

        r1 = TokenizedReport(
            report_id=1, creation_time=datetime.now(), status="RESOLVED",
            component="foo", dupe_of=None, summary=text1, comments=[],
            text=text1+" foo", tokens=tokenizer.tokenize(text1+" foo")
        )
        r2 = TokenizedReport(
            report_id=2, creation_time=datetime.now(), status="RESOLVED",
            component="foo", dupe_of=1, summary=text2, comments=[],
            text=text2+" foo", tokens=tokenizer.tokenize(text2+" foo")
        )
        r3 = TokenizedReport(
            report_id=3, creation_time=datetime.now(), status="RESOLVED",
            component="Address Bar", dupe_of=None, summary=text3, comments=[],
            text=text3+" Address Bar", tokens=tokenizer.tokenize(text3+" Address Bar")
        )
        r4 = TokenizedReport(
            report_id=4, creation_time=datetime.now(), status="RESOLVED",
            component="Artifacts", dupe_of=None, summary=text4, comments=[],
            text=text4+" Artifacts", tokens=tokenizer.tokenize(text4+" Artifacts")
        )
        for r in [r1, r2, r3, r4]:
            create_tokenized_report(r.report_id, r.creation_time, r.status, r.component, r.dupe_of, r.summary, r.comments, r.text, r.tokens)

        rep = TokenizedReport(
            report_id=5, creation_time=datetime.now(), status="NEW",
            component="Address Bar", dupe_of=None, summary=report, comments=[],
            text=report, tokens=tokenizer.tokenize(report+" Address Bar")
        )

        test_db_reports = get_tokenized_reports(limit=50)

        similar_reports = clusterizer.clusterize(rep, test_db_reports)
        possible_duplicates = classifier.get_possible_duplicates(rep, similar_reports)
        self.assertIn(r3.report_id, possible_duplicates['report_id'].values)


class TestSIDRD(BaseTest):
    """TestCase class for SIDRD duplicate detection funcionalities"""

    def test_get_duplicates(self):
        sidrd = SIDRD()
        tokenizer = sidrd.tokenizer

        text1 = "(This is the test 1). 8462 : <Also know as TEST1 or test_1.js>. Do not google it in https://google.com :)"
        text2 = "I googled test_1.js and there was no way i know it is a test"
        text3 = "New tab address bar freezes until mouse is moved"
        text4 = "Small typo in artifacts.py:1:16. No need to pull request now"
        report = "Address bar freezes when making new tabs"

        r1 = TokenizedReport(
            report_id=1, creation_time=datetime.now(), status="RESOLVED",
            component="foo", dupe_of=None, summary=text1, comments=[],
            text=text1+" foo", tokens=tokenizer.tokenize(text1+" foo")
        )
        r2 = TokenizedReport(
            report_id=2, creation_time=datetime.now(), status="RESOLVED",
            component="foo", dupe_of=1, summary=text2, comments=[],
            text=text2+" foo", tokens=tokenizer.tokenize(text2+" foo")
        )
        r3 = TokenizedReport(
            report_id=3, creation_time=datetime.now(), status="RESOLVED",
            component="Address Bar", dupe_of=None, summary=text3, comments=[],
            text=text3+" Address Bar", tokens=tokenizer.tokenize(text3+" Address Bar")
        )
        r4 = TokenizedReport(
            report_id=4, creation_time=datetime.now(), status="RESOLVED",
            component="Artifacts", dupe_of=None, summary=text4, comments=[],
            text=text4+" Artifacts", tokens=tokenizer.tokenize(text4+" Artifacts")
        )
        for r in [r1, r2, r3, r4]:
            create_tokenized_report(r.report_id, r.creation_time, r.status, r.component, r.dupe_of, r.summary, r.comments, r.text, r.tokens)

        rep = TokenizedReport(
            report_id=5, creation_time=datetime.now(), status="NEW",
            component="Address Bar", dupe_of=None, summary=report, comments=[],
            text=None, tokens=None
        )

        test_db_reports = get_tokenized_reports(limit=50)

        report, duplicates = sidrd.get_duplicates(rep, test_db_reports)
        self.assertIn(r3.report_id, [dup['report_id'] for dup in duplicates])
        self.assertIsNotNone(report.text)
        self.assertIsNotNone(report.tokens)
