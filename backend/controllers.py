"""Module for controllers."""
from datetime import datetime

from models import Report, TokenizedReport
from exceptions import ReportAlreadyExists

# CRUD Reports

def get_report(report_id: int) -> Report:
    """
    Get a report from the database.
    Args:
        report_id: id of the report.
    Returns:
        report:Report report from the database.
    Exceptions:
        ReportNotFound. If the report does not exist.
    Example:
        >>> report = get_report(12345)
    """
    report = Report.get(report_id) # May raise ReportNotFound
    return report

def get_reports(filters: dict = {}, limit: int = 100) -> list:
    """
    Get limit reports from the database.
    Args:
        filters: dict, filters to apply to the reports.
            Available fields: report_id, dupe_of, status, component, creation_time
            Must be strings
        limit: limit of the number of reports to get.
    Returns:
        reports: list of reports (Report)
    Exceptions:
        NoReportsFound. If there are no reports in the database.
    Example:
        >>> reports = get_reports(filters={"dupe_of": "None"}, limit=10)
    """
    reports = Report.get_reports(filters, limit) # May raise NoReportsFound
    return reports

def create_report(report_id: int, creation_time: datetime,
                status: str, component: str, dupe_of: int,
                summary: str, comments: list) -> Report:
    """
    Create a report in the database.
    Args:
        report_id: id of the report.
        creation_time: creation time of the report.
        status: status of the report.
        component: component of the report.
        dupe_of: id of the report this is a duplicate of. None if not a duplicate.
        summary: summary of the report.
        comments: list of comments for the report.
    Returns:
        report:Report report from the database.
    Exceptions:
        ReportAlreadyExists. If the report already exists.
    Example:
        >>> report = create_report(report)
    """
    report = Report.insert(report_id, creation_time, 
                        status, component, dupe_of, 
                        summary, comments) # May raise ReportAlreadyExists
    return report

def create_many_reports(reports: list) -> int:
    """
    Create many reports in the database.
    Args:
        reports: list of reports (Report)
    Returns:
        reports: number of reports created.
    Example:
        >>> reports = create_many_reports([{'report_id': 1, 'creation_time': ...}, ...])
    """
    inserted_reports = 0

    if reports:
        for rp in reports:
            report = Report(**rp) if type(rp) == dict else rp
            try:
                create_report(report.report_id, report.creation_time, 
                                report.status, report.component, report.dupe_of, 
                                report.summary, report.comments) # May raise ReportAlreadyExists
                inserted_reports += 1
            except ReportAlreadyExists:
                pass
    
    return inserted_reports

def update_report(_id: int, report_id: int, creation_time: datetime,
                status: str, component: str, dupe_of: int,
                summary: str, comments: list, text: str, tokens: list) -> Report:
    """
    Update a report in the database.
    Args:
        _id: mongo id of the report.
        report_id: id of the report.
        creation_time: creation time of the report.
        status: status of the report.
        component: component of the report.
        dupe_of: id of the report this is a duplicate of. None if not a duplicate.
        summary: summary of the report.
        comments: list of comments for the report.
        text: text of the report.
        tokens: list of tokens for the report.
    Returns:
        report:Report report from the database.
    Exceptions:
        ReportNotFound. If the report does not exist.
    Example:
        >>> report = update_report(12345, 657,...)
    """
    report = TokenizedReport.update(_id, report_id, creation_time, 
                        status, component, dupe_of, 
                        summary, comments, text, tokens) # May raise ReportNotFound
    return report

def delete_report(report_id: int) -> None:
    """
    Delete a report from the database.
    Args:
        report_id: report_id of the report.
    Exceptions:
        ReportNotFound. If the report does not exist.
    Example:
        >>> delete_report(12345)
    """
    return Report.delete(report_id=report_id) # May raise ReportNotFound

def delete_all_reports() -> int:
    """
    Delete all reports from the database.
    Returns:
        int: number of reports deleted.
    Example:
        >>> num_deleted = delete_all_reports()
    """
    return Report.delete_all() # May raise NoReportsFound

###############################################################################

# SIDRD Update


def get_tokenized_report(report_id: int) -> TokenizedReport:
    """
    Get a report from the database.
    Args:
        report_id: id of the report.
    Returns:
        report:TokenizedReport report from the database.
    Exceptions:
        ReportNotFound. If the report does not exist.
    Example:
        >>> report = get_tokenized_report(12345)
    """
    report = TokenizedReport.get(report_id) # May raise ReportNotFound
    return report

def get_tokenized_reports(filters: dict = {}, limit: int = 5000) -> list:
    """
    Get all tokenized reports from the database.
    Args:
        limit: limit of the number of reports to get.
    Returns:
        reports: list of reports (TokenizedReport)
    Exceptions:
        NoReportsFound. If there are no reports in the database.
    Example:
        >>> reports = get_tokenized_reports()
    """
    reports = TokenizedReport.get_reports(filters=filters, limit=limit) # May raise NoReportsFound
    return reports


def create_tokenized_report(report_id: int, creation_time: datetime,
                status: str, component: str, dupe_of: int,
                summary: str, comments: list, text:str, tokens:list) -> TokenizedReport:
    """
    Create a report in the database.
    Args:
        report_id: id of the report.
        creation_time: creation time of the report.
        status: status of the report.
        component: component of the report.
        dupe_of: id of the report this is a duplicate of. None if not a duplicate.
        summary: summary of the report.
        comments: list of comments for the report.
        text: text of the report.
        tokens: list of tokens for the report.
    Returns:
        report:TokenizedReport report from the database.
    Exceptions:
        ReportAlreadyExists. If the report already exists.
    Example:
        >>> report = create_tokenized_report(report)
    """
    report = TokenizedReport.insert(report_id, creation_time, 
                        status, component, dupe_of, 
                        summary, comments, text, tokens) # May raise ReportAlreadyExists
    return report

def create_many_tokenized_reports(reports: list) -> int:
    """
    Create many reports in the database.
    Args:
        reports: list of reports (TokenizedReport)
    Returns:
        reports: number of reports created.
    Example:
        >>> reports = create_many_tokenized_reports([{'report_id': 1, 'creation_time': ...}, ...])
    """
    inserted_reports = 0

    if reports:
        for rp in reports:
            report = TokenizedReport(**rp) if type(rp) == dict else rp
            try:
                create_tokenized_report(report.report_id, report.creation_time, 
                                report.status, report.component, report.dupe_of, 
                                report.summary, report.comments, report.text, report.tokens) # May raise ReportAlreadyExists
                inserted_reports += 1
            except ReportAlreadyExists:
                pass
    
    return inserted_reports

def update_tokenized_report(_id: int, report_id: int, creation_time: datetime,
                status: str, component: str, dupe_of: int,
                summary: str, comments: list, text: str, tokens: list) -> TokenizedReport:
    """
    Update a report in the database.
    Args:
        _id: mongo id of the report.
        report_id: id of the report.
        creation_time: creation time of the report.
        status: status of the report.
        component: component of the report.
        dupe_of: id of the report this is a duplicate of. None if not a duplicate.
        summary: summary of the report.
        comments: list of comments for the report.
        text: text of the report.
        tokens: list of tokens for the report.
    Returns:
        report:TokenizedReport report from the database.
    Exceptions:
        ReportNotFound. If the report does not exist.
    Example:
        >>> report = update_report(12345, 657,...)
    """
    report = TokenizedReport.update(_id, report_id, creation_time, 
                        status, component, dupe_of, 
                        summary, comments, text, tokens) # May raise ReportNotFound
    return report

def delete_tokenized_report(report_id: int) -> None:
    """
    Delete a report from the database.
    Args:
        report_id: report_id of the report.
    Exceptions:
        ReportNotFound. If the report does not exist.
    Example:
        >>> delete_tokenized_report(12345)
    """
    return TokenizedReport.delete(report_id=report_id) # May raise ReportNotFound

def delete_all_tokenized_reports() -> int:
    """
    Delete all reports from the database.
    Returns:
        int: number of reports deleted.
    Example:
        >>> num_deleted = delete_all_tokenized_reports()
    """
    return TokenizedReport.delete_all() # May raise NoReportsFound
