"""Api exceptions list"""

# General
class BaseException(Exception):
    """Base class for Api exceptions"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class MissingParameters(BaseException):
    """Raised when one or more parameters are missing"""
    def __init__(self, missing_params: list):
        super().__init__(f"Missing parameters: {missing_params}")

class InvalidParameters(BaseException):
    """Raised when one or more parameters are invalid"""
    def __init__(self, invalid_params: list):
        super().__init__(f"Invalid parameters: {invalid_params}")

# Reports
class NoReportsFound(BaseException):
    """
    Exception raised when no reports are found in the database.
    """
    def __init__(self):
        super().__init__("No reports found")

class ReportNotFound(BaseException):
    """
    Exception raised when a report is not found
    
    Args:
        report_id: id of the report
    """
    def __init__(self, report_id: str):
        super().__init__(f"Report '{report_id}' not found")

class ReportAlreadyExists(BaseException):
    """
    Exception raised when a report already exists (duplicated report_id)
    
    Args:
        report_id: id of the report
    """
    def __init__(self, report_id: str):
        super().__init__(f"Report with report_id '{report_id}' already exists")
