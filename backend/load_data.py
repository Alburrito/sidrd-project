import pandas
from models import TokenizedReport
from pymongo import MongoClient
import numpy as np

df = pandas.read_pickle('notebooks/data/preprocessed_reports.pkl')

client = MongoClient('localhost', 27017)
db = client['reports_db']
reports_collection = db['tokenized_bug_reports']

for i, row in df.iterrows():
    report = {
        'report_id': row['report_id'],
        'creation_time': row['creation_time'],
        'dupe_of': row['dupe_of'],
        'status': row['status'],
        'component': row['component'],
        'summary': row['summary'],
        'comments': row['comments'],
        'text': row['text2'],
        'tokens': row['tokens2S']
    }
    reports_collection.insert_one(report)