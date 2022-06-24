"""Module for reports related models."""
from os import environ
from datetime import datetime
from pymongo import MongoClient

from exceptions import NoReportsFound, ReportAlreadyExists, ReportNotFound

# CONSTANTS
BUG_REPORTS_COLLECTION = "bug_reports"

# ENV VARIABLES
HOST = environ.get("DB_HOST", 'localhost')
PORT = int(environ.get("DB_PORT", '27017'))
DB_NAME = environ.get("DB_NAME", "bug_reports_db")

# CLIENT AND DATABASE CONNECTION
client = MongoClient(HOST, PORT)
conn_db = client[DB_NAME]

# COLLECTIONS
if BUG_REPORTS_COLLECTION in conn_db.list_collection_names():
    reports_collection = conn_db[BUG_REPORTS_COLLECTION]
else:
    reports_collection = conn_db.create_collection(BUG_REPORTS_COLLECTION)


class Report():
    """Class for reports.
    
    Attributes:
        _id: mongoDB generated id.
        report_id: int, unique id of the report.
        creation_time: datetime, date of creation.
        status: str, status of the report.
        component: str, component of the report.
        dupe_of: int, id of the report this is a duplicate of. None if not a duplicate.
        summary: str, summary of the report.
        comments: list, list of comments on the report.
    """

    def __init__(self, 
                report_id: int, creation_time: datetime,
                status: str, component: str, dupe_of: int,
                summary: str, comments: list, _id: int = None):
        # mongoDB will generate an extra field self._id when created
        self._id = _id
        self.report_id = report_id
        self.creation_time = creation_time
        self.status = status
        self.component = component
        self.dupe_of = dupe_of
        self.summary = summary
        self.comments = comments

    def __str__(self):
        return f"Report {self.report_id} (dupe of {self.dupe_of}) - {self.summary}"
    
    @classmethod
    def get(cls, report_id: int) -> 'Report':
        """
        Get a report from the database.
        Args:
            report_id: id of the report.
        Returns:
            report:Report report from the database.
        Exceptions:
            ReportNotFound. If the report does not exist.
        Example:
            >>> report = Report.get(12345)
        """
        report = reports_collection.find_one({"report_id": report_id})
        if report is None:
            raise ReportNotFound(report_id)
        
        return cls(**report)

    @classmethod
    def get_reports(cls, limit: int = 100) -> list:
        """
        Get limit reports from the database.
        Args:
            limit: limit of the number of reports to get.
        Returns:
            reports: list of reports (Report)
        Exceptions:
            NoReportsFound. If there are no reports in the database.
        Example:
            >>> reports = Report.get_reports(limit=10)
        """
        # TODO: incorporate filters (only master?)

        reports = reports_collection.find({}).sort(
            "creation_time", -1
        ).limit(limit)

        if reports is None:
            raise NoReportsFound()

        return [cls(**report) for report in reports]
    
    @classmethod
    def insert(cls,
            report_id: int, creation_time: datetime,
            status: str, component: str, dupe_of: int,
            summary: str, comments: list) -> 'Report':
        """
        Create a new report in the database.

        Returns:
            The resulting Report object.
        Exceptions:
            ReportAlreadyExists. If a report with the same report_id already exists.
        Example:
            >>> report = Report.insert(123, datetime.now(), "DUPLICATED", "component", 23, "summary", ["com1", "com2"...])
        """
        try:
            report = cls.get(report_id)
            if report is not None:
                raise ReportAlreadyExists(report_id)
        except ReportNotFound:
            report = reports_collection.insert_one({
                "report_id": report_id,
                "creation_time": creation_time,
                "status": status,
                "component": component,
                "dupe_of": dupe_of,
                "summary": summary,
                "comments": comments
            })
            report = cls(
                _id = report.inserted_id,
                report_id=report_id,
                creation_time=creation_time,
                status=status,
                component=component,
                dupe_of=dupe_of,
                summary=summary,
                comments=comments
            )
        return report


    @classmethod
    def update(cls) -> 'Report':
        pass

    @classmethod
    def delete(cls) -> int:
        pass

    @classmethod
    def delete_all(cls) -> int:
        """
        Delete all reports from the database.
        Returns:
            The number of reports deleted.
        Exceptions:
            NoReportsFound. If there are no reports in the database.
        Example:
            >>> num_deleted = Report.delete_all()
        """
        num_deleted = reports_collection.delete_many({}).deleted_count
        if num_deleted == 0:
            raise NoReportsFound()

        return num_deleted
