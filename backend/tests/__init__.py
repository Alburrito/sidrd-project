"""Base test module for the project."""
from datetime import datetime
import json
import unittest
from os import environ
from pymongo import MongoClient

HOST = environ.get("TEST_DB_HOST", 'localhost')
PORT = int(environ.get("TEST_DB_PORT", '27017'))
DB_NAME = environ.get("TEST_DB_NAME", "testing_bug_reports_db")
REPORTS_COLLECTION = environ.get("TEST_DB_REPORT_COLLECTION", "testing_bug_reports")
TOKENIZED_COLLECTION = environ.get("TEST_DB_TOKENIZED_COLLECTION", "testing_tokenized_bug_reports")
environ['DB_HOST'] = HOST
environ['DB_PORT'] = str(PORT)
environ['DB_NAME'] = DB_NAME
environ['DB_REPORT_COLLECTION'] = REPORTS_COLLECTION
environ['DB_TOKENIZED_COLLECTION'] = TOKENIZED_COLLECTION

class BaseTest(unittest.TestCase):
    """Base test class for the project."""

    def setUp(self):
        """
        Setup the test.
        Initializes mongoDB test database with test data.
        This test data consists on:
        - Three duplicate reports
        - Their three master reports
        """
        self.client = MongoClient(HOST, PORT)
        self.conn_db = self.client[DB_NAME]
        self.reports_collection = self.conn_db[REPORTS_COLLECTION]
        self.tokenized_collection = self.conn_db[TOKENIZED_COLLECTION]

        with open(f"tests/{REPORTS_COLLECTION}.json", "r") as f:
            test_data = json.load(f)
        with open(f"tests/{TOKENIZED_COLLECTION}.json", "r") as f:
            tokenized_data = json.load(f)
        
        for report in test_data:
            del report['_id']
            report['creation_time'] = datetime.strptime(
                report['creation_time']['$date'],
                "%Y-%m-%dT%H:%M:%SZ"
            )
            self.reports_collection.insert_one(report)
        
        for report in tokenized_data:
            del report['_id']
            report['creation_time'] = datetime.strptime(
                report['creation_time']['$date'],
                "%Y-%m-%dT%H:%M:%SZ"
            )
            self.tokenized_collection.insert_one(report)

    def tearDown(self):
        """Teardown the test."""
        self.reports_collection.drop()
        self.tokenized_collection.drop()
        self.client.drop_database(DB_NAME)
        self.client.close()
