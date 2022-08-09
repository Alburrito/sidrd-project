"""Module to test report related functionalities."""
from datetime import datetime

import controllers as c
from exceptions import ReportAlreadyExists, ReportNotFound, NoReportsFound
from tests import BaseTest

class TestReports(BaseTest):
    """TestCase class to test the report related functionalities"""

    def test_get_report(self):
        """Test get_report function"""
        report_id = 1606814
        report = c.get_report(report_id)
        self.assertEqual(report.report_id, report_id)
        self.assertRaises(ReportNotFound, c.get_report, -1)


    def test_get_reports(self):
        """Test get_reports function"""
        reports = c.get_reports()
        self.assertGreater(len(reports), 0)

        reports = c.get_reports(
            filters={"dupe_of": None},
            limit=2
        )
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0].dupe_of, None)
        self.assertEqual(reports[1].dupe_of, None)
        self.assertRaises(NoReportsFound, c.get_reports, filters={"dupe_of": -1})


    def test_create_report(self):
        """Test create_report function"""
        report = c.create_report(1, datetime.now(), "DUPLICATE",
                    "component", 23, "summary", 
                    {'1': {"comments": [
                        {"text": "comment1"}, {"text": "comment2"}
                    ]}}
                )
        self.assertEqual(report.report_id, 1)

        report = c.create_report(23, datetime.now(), "RESOLVED",
            "component", None, "summary", 
            {'23': {"comments": [
                {"text": "comment1"}, {"text": "comment2"}
            ]}}
        )
        self.assertEqual(report.dupe_of, None)
        self.assertRaises(ReportAlreadyExists, c.create_report, 23, datetime.now(), "DUPLICATE", "component", None, "summary", {'1': {} })


    def test_create_many_reports(self):
        """Test create_many_reports function"""
        reports = [
            {"report_id": 1, "creation_time": datetime.now(), "status": "DUPLICATE", "component": "component", "dupe_of": 23, "summary": "summary", "comments": [{"text": "comment1"}, {"text": "comment2"}]},
            {"report_id": 23, "creation_time": datetime.now(), "status": "RESOLVED", "component": "component", "dupe_of": None, "summary": "summary", "comments": [{"text": "comment1"}, {"text": "comment2"}]}
        ]
        inserted_reports = c.create_many_reports(reports)
        self.assertEqual(inserted_reports, 2)
        self.assertEqual(c.get_report(1).dupe_of, 23)
        self.assertEqual(c.get_report(23).dupe_of, None)


    def test_delete_report(self):
        """Test delete_report function"""
        reports_number = len(c.get_reports())
        report = c.create_report(160, datetime.now(), "DUPLICATE",
                    "component", 23, "summary", 
                    {'160': {"comments": [
                        {"text": "comment1"}, {"text": "comment2"}
                    ]}}
                )
        self.assertEqual(len(c.get_reports()), reports_number + 1)
        c.delete_report(report.report_id)
        self.assertEqual(len(c.get_reports()), reports_number)
        self.assertRaises(ReportNotFound, c.get_report, report.report_id)
    

    def test_delete_all_reports(self):
        """Test delete_all_reports function"""
        self.assertGreater(len(c.get_reports()), 0)
        c.delete_all_reports()
        self.assertRaises(NoReportsFound, c.get_reports)
        self.assertRaises(NoReportsFound, c.delete_all_reports)
