"""Module for reports related models."""
from lib2to3.pgen2 import token
from os import environ
from datetime import datetime
from pymongo import MongoClient

from exceptions import NoReportsFound, ReportAlreadyExists, ReportNotFound

# CONSTANTS

# ENV VARIABLES
HOST = environ.get("DB_HOST", 'localhost')
PORT = int(environ.get("DB_PORT", '27017'))
DB_NAME = environ.get("DB_NAME", "bug_reports_db")
BUG_REPORTS_COLLECTION = environ.get("DB_REPORT_COLLECTION", "bug_reports")
TOKENIZED_COLLECTION = environ.get("DB_TOKENIZED_COLLECTION", "tokenized_bug_reports")

# CLIENT AND DATABASE CONNECTION
client = MongoClient(HOST, PORT)
conn_db = client[DB_NAME]

# COLLECTIONS
if BUG_REPORTS_COLLECTION in conn_db.list_collection_names():
    reports_collection = conn_db[BUG_REPORTS_COLLECTION]
else:
    reports_collection = conn_db.create_collection(BUG_REPORTS_COLLECTION)

if TOKENIZED_COLLECTION in conn_db.list_collection_names():
    tokenized_collection = conn_db[TOKENIZED_COLLECTION]
else:
    tokenized_collection = conn_db.create_collection(TOKENIZED_COLLECTION)


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

    INTEGER_FIELDS = ["report_id", "dupe_of"]
    DATETIME_FIELDS = ["creation_time"]
    STRING_FIELDS = ["status", "component", "summary"]

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
    def get_reports(cls, filters: dict = {}, limit: int = 100) -> list:
        """
        Get reports from the database filtered by the filters.
        Args:
            filters: dict, filters to apply to the reports.
                Available fields: report_id, dupe_of, status, component, creation_time
                Must be strings
            limit: number of reports to get.
        Returns:
            reports: list of reports (Report)
        Exceptions:
            NoReportsFound. If there are no reports in the database.
        Example:
            >>> reports = Report.get_reports(filters={"dupe_of": "None"}, limit=10)
        """
        q_filters = {}

        for key, value in filters.items():
            try:
                if key in cls.INTEGER_FIELDS:
                    q_filters[key] = int(value)
                elif key in cls.DATETIME_FIELDS:
                    q_filters[key] = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                elif key in cls.STRING_FIELDS:
                    q_filters[key] = str(value)
                else:
                    q_filters[key] = value
            except Exception:
                q_filters[key] = value
        
        reports = list(reports_collection.find(q_filters).sort(
            "creation_time", -1
        ).limit(limit))

        if reports == []:
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
            >>> report = Report.insert(123, datetime.now(), "DUPLICATE", "component", 23, "summary", ["com1", "com2"...])
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
    def create_many(cls, reports: list) -> int:
        """
        Create many reports in the database.
        Args:
            reports: list of reports to create. (dicts)
        Returns:
            The number of reports created.
        """
        to_insert = [cls(**report) for report in reports]
        return reports_collection.insert_many(to_insert).inserted_ids

    @classmethod
    def update(cls) -> 'Report':
        pass

    @classmethod
    def delete(cls, report_id: int) -> int:
        """
        Delete a report from the database.
        Args:
            report_id: report_id of the report to delete. (not mongoDB id)
        Returns:
            The number of reports deleted.
        Exceptions:
            ReportNotFound. If the report does not exist.
        Example:
            >>> num_reports = Report.delete(12345)
        """
        num_deleted = reports_collection.delete_one({"report_id": report_id})
        if num_deleted == 0:
            raise ReportNotFound(report_id)

        return num_deleted
        

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

###############################################################################################

# SIDRD Update

class TokenizedReport(Report):

    def __init__(self, text, tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        self.tokens = tokens

    @classmethod
    def get(cls, report_id: int) -> 'TokenizedReport':
        """
        Get a report from the database.
        Args:
            report_id: id of the report.
        Returns:
            report:TokenizedReport report from the database.
        Exceptions:
            ReportNotFound. If the report does not exist.
        Example:
            >>> report = TokenizedReport.get(12345)
        """
        report = tokenized_collection.find_one({"report_id": report_id})
        if report is None:
            raise ReportNotFound(report_id)
        
        return cls(**report)
    

    @classmethod
    def get_reports(cls, filters: dict = {}, limit: int = 100) -> list:
        """
        Get reports from the database filtered by the filters.
        Args:
            filters: dict, filters to apply to the reports.
                Available fields: report_id, dupe_of, status, component, creation_time
                Must be strings
            limit: number of reports to get.
        Returns:
            reports: list of reports (TokenizedReport)
        Exceptions:
            NoReportsFound. If there are no reports in the database.
        Example:
            >>> reports = TokenizedReport.get_reports(filters={"dupe_of": "None"}, limit=10)
        """
        q_filters = {}

        for key, value in filters.items():
            try:
                if key in cls.INTEGER_FIELDS:
                    q_filters[key] = int(value)
                elif key in cls.DATETIME_FIELDS:
                    q_filters[key] = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                elif key in cls.STRING_FIELDS:
                    q_filters[key] = str(value)
                else:
                    q_filters[key] = value
            except Exception:
                q_filters[key] = value
        
        reports = list(tokenized_collection.find(q_filters).sort(
            "creation_time", -1
        ).limit(limit))

        if reports == []:
            raise NoReportsFound()

        return [cls(**report) for report in reports]

    @classmethod
    def insert(cls,
            report_id: int, creation_time: datetime,
            status: str, component: str, dupe_of: int,
            summary: str, comments: list, text: str,
            tokens: list) -> 'TokenizedReport':
        """
        Create a new report with tokens in the database.

        Returns:
            The resulting TokenizedReport object.
        Exceptions:
            ReportAlreadyExists. If a report with the same report_id already exists.
        Example:
            >>> report = TokenizedReport.insert(
                    123, datetime.now(), "DUPLICATE", "component", 23, "summary", 
                    ["com1", "com2"...], "text", ["token1", "token2"...]
                )
        """
        try:
            report = cls.get(report_id)
            if report is not None:
                raise ReportAlreadyExists(report_id)
        except ReportNotFound:
            report = tokenized_collection.insert_one({
                "report_id": report_id,
                "creation_time": creation_time,
                "status": status,
                "component": component,
                "dupe_of": dupe_of,
                "summary": summary,
                "comments": comments,
                "text": text,
                "tokens": tokens
            })
            report = cls(
                _id = report.inserted_id,
                report_id=report_id,
                creation_time=creation_time,
                status=status,
                component=component,
                dupe_of=dupe_of,
                summary=summary,
                comments=comments,
                text=text,
                tokens=tokens
            )
        return report

    @classmethod
    def create_many(cls, reports: list) -> int:
        """
        Create many reports in the database.
        Args:
            reports: list of reports to create. (dicts)
        Returns:
            The number of reports created.
        """
        to_insert = [cls(**report) for report in reports]
        return tokenized_collection.insert_many(to_insert).inserted_ids

    @classmethod
    def update(cls) -> 'TokenizedReport':
        pass

    @classmethod
    def delete(cls, report_id: int) -> int:
        """
        Delete a report from the database.
        Args:
            report_id: report_id of the report to delete. (not mongoDB id)
        Returns:
            The number of reports deleted.
        Exceptions:
            ReportNotFound. If the report does not exist.
        Example:
            >>> num_reports = TokenizedReport.delete(12345)
        """
        num_deleted = tokenized_collection.delete_one({"report_id": report_id})
        if num_deleted == 0:
            raise ReportNotFound(report_id)

        return num_deleted
        

    @classmethod
    def delete_all(cls) -> int:
        """
        Delete all reports from the database.
        Returns:
            The number of reports deleted.
        Exceptions:
            NoReportsFound. If there are no reports in the database.
        Example:
            >>> num_deleted = TokenizedReport.delete_all()
        """
        num_deleted = tokenized_collection.delete_many({}).deleted_count
        if num_deleted == 0:
            raise NoReportsFound()

        return num_deleted
