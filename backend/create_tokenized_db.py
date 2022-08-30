from pymongo import MongoClient
from datetime import datetime
import json
from sidrd import Tokenizer
from tests import REPORTS_COLLECTION

DB_NAME="TEST_bug_reports_db"
# REPORTS_COLLECTION="bug_reports"
# TOKENS_COLLECTION="tokenized_bug_reports"
REPORTS_COLLECTION="TMP_testing_bug_reports copy"
TOKENS_COLLECTION="testing_tokenized_bug_reports"
HOST="localhost"
PORT=27017

client = MongoClient(HOST, PORT)
conn_db = client[DB_NAME]
reports_collection = conn_db[REPORTS_COLLECTION]
tokenized_reports_collection = conn_db[TOKENS_COLLECTION]

with open(f"tests/{REPORTS_COLLECTION}.json", "r") as f:
    data = json.load(f)


tokenizer = Tokenizer()

for report in data:
    del report['_id']
    report['creation_time'] = datetime.strptime(
        report['creation_time']['$date'],
        "%Y-%m-%dT%H:%M:%SZ"
    )
    report['text'] = report['summary'] + ' ' + report['component']
    report['tokens'] = tokenizer.tokenize(report['text'], 'stem')
    tokenized_reports_collection.insert_one(report)