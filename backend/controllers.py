"""Module for controllers."""
import os
from bugzilla import Bugzilla
from datetime import datetime

from models import Report

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
    # TODO: change to mongo id
    report = Report.get(report_id) # May raise ReportNotFound
    return report

def get_reports(limit: int = 100) -> list:
    """
    Get limit reports from the database.
    Args:
        limit: limit of the number of reports to get.
    Returns:
        reports: list of reports (Report)
    Exceptions:
        NoReportsFound. If there are no reports in the database.
    Example:
        >>> reports = get_reports(limit=10)
    """
    reports = Report.get_reports(limit=limit) # May raise NoReportsFound
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
    for report in reports:
        create_report(report.report_id, report.creation_time, 
                        report.status, report.component, report.dupe_of, 
                        report.summary, report.comments) # May raise ReportAlreadyExists
    return reports

def update_report(_id: int, report_id: int, creation_time: datetime,
                status: str, component: str, dupe_of: int,
                summary: str, comments: list) -> Report:
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
    Returns:
        report:Report report from the database.
    Exceptions:
        ReportNotFound. If the report does not exist.
    Example:
        >>> report = update_report(12345, 657,...)
    """
    report = Report.update(_id, report_id, creation_time, 
                        status, component, dupe_of, 
                        summary, comments) # May raise ReportNotFound
    return report

def delete_report(_id: int) -> None:
    """
    Delete a report from the database.
    Args:
        _id: mongo id of the report.
    Exceptions:
        ReportNotFound. If the report does not exist.
    Example:
        >>> delete_report(12345)
    """
    Report.delete(_id) # May raise ReportNotFound

def delete_all_reports() -> int:
    """
    Delete all reports from the database.
    Returns:
        int: number of reports deleted.
    Example:
        >>> num_deleted = delete_all_reports()
    """
    return Report.delete_all() # May raise NoReportsFound


# Bugzilla scraper

BUGZILLA_API_KEY = os.environ.get('BUGZILLA_API_KEY', '')
BUGZILLA_BASE_URL = "https://bugzilla.mozilla.org/rest/"


def parse_report(scraped_report: dict) -> Report:
    """
    Parse a report from the Bugzilla API.
    Args:
        scraped_report: report from the Bugzilla API.
    Returns:
        report:Report report from the database.
    Example:
        >>> report = parse_report(scraped_report)
    """
    report_id = int(scraped_report['id'])
    creation_time = datetime.strptime(scraped_report['creation_time'], 
                                    '%Y-%m-%dT%H:%M:%SZ')
    status = scraped_report['status']
    component = scraped_report['component']
    dupe_of = int(scraped_report['dupe_of']) if scraped_report['dupe_of'] else None
    summary = scraped_report['summary']
    comments = scraped_report['comments'][:1] if len(scraped_report['comments']) > 0 else []

    return Report(report_id, creation_time, status, 
                component, dupe_of, summary, comments)

def scrape_reports(product: str, limit: int, mode: str,
                from_date: datetime) -> dict:
    """
    Scrapes bugzilla for reports and stores them in the database.
    Args:
        product: product to scrape.
        limit: limit of the number of reports TO SCRAPE.
        mode: mode of the scraper. 
            Can be 'mixed' (for duplicated and their masters) or 'master' (only masters)
            ATTENTION: 'mixed' mode will create more than limit reports.
        from_date: reports must have been created after this date.
    Returns:
        dict: {'master': int, 'duplicate': int} 
        Number of master and duplicate reports scraped.
    Exceptions:
        BaseException. If an error occurs.
    Example:
        >>> num_reports = scrape_reports("firefox", 50, "mixed",...)
    """
    bugzilla = Bugzilla(BUGZILLA_BASE_URL, api_key=BUGZILLA_API_KEY)

    terms = [
        {'product': product},
        {'limit': str(limit)},
    ]

    # Need string date format 2000-04-06T04:22:59Z
    try:
        creation_time = f'{datetime.strftime(from_date, "%Y-%m-%dT%H:%M:%SZ")}'
    except ValueError:
        raise BaseException() # TODO: better error handling
    terms['creation_time'] = creation_time

    if mode == 'mixed':
        terms['resultion'] = 'DUPLICATE'
    elif mode == 'master':
        terms['resolution'] = 'FIXED'
        terms['dupe_of'] = "None"
    else:
        raise BaseException() # TODO: change to other exception

    try:
        result = bugzilla.search_bugs(terms)
        scraped_reports = result['bugs']
    except Exception as e:
        raise e

    parsed_reports = [parse_report(scraped_report) for scraped_report in scraped_reports]
    if parsed_reports:
        num_reports = create_many_reports(parsed_reports)

    return num_reports
