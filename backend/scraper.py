"""Module for scraping interface"""
import os
import sys
import csv
from datetime import datetime
from bugzilla import Bugzilla

from controllers import create_many_reports, delete_report
from models import Report

BUGZILLA_API_KEY = os.environ.get('BUGZILLA_API_KEY', '')
BUGZILLA_BASE_URL = "https://bugzilla.mozilla.org/rest/"

class Scraper():
    """
    Class for scraping interface.
    """
    def __init__(self):
        """
        Initialize the scraper.
        """
        self.bugzilla = Bugzilla(BUGZILLA_BASE_URL, api_key=BUGZILLA_API_KEY)

        
    def _parse_report(self, scraped_report: dict) -> Report:
        """
        Parse a report from the Bugzilla API.
        Args:
            scraped_report: report from the Bugzilla API.
        Returns:
            report:Report report from the database.
        Example:
            >>> report = self._parse_report(scraped_report)
        """
        report_id = int(scraped_report['id'])
        creation_time = datetime.strptime(scraped_report['creation_time'], 
                                        '%Y-%m-%dT%H:%M:%SZ')
        status = scraped_report['status']
        component = scraped_report['component']
        dupe_of = int(scraped_report['dupe_of']) if scraped_report['dupe_of'] else None
        summary = scraped_report['summary']
        comments = self._scrape_comments(report_id)
        comments = comments[:1] if len(comments) > 1 else []

        return Report(report_id, creation_time, status, 
                    component, dupe_of, summary, comments)
                    

    def _scrape_reports(self, terms: list) -> list:
        """
        Scrape Bugzilla for reports.
        Args:
            terms: list of terms to search for.
        Returns:
            reports: list of reports (Report)
        """
        try:
            scraped_reports = self.bugzilla.search_bugs(terms)
            parsed_reports = [self._parse_report(scraped_report) for scraped_report in scraped_reports['bugs']]
            return parsed_reports
        except Exception as e:
            return None

    def _scrape_report(self, report_id: int) -> Report:
        """
        Scrape Bugzilla for a single report.
        Args:
            report_id: id of the report to scrape.
        Returns:
            report:Report report from the database.
        Example:
            >>> report = self._scrape_report(12345)
        """
        try:
            scraped_report = self.bugzilla.get_bug(report_id)
            return self._parse_report(scraped_report)
        except Exception as e:
            return None
    
    def _scrape_master_report(self, report_id: int) -> Report:
        """
        Scrape Bugzilla for the master of a given report.
        If the retrieved report is duplicated, returns the corresponding master.
        Args:
            report_id: id of the report to scrape.
        Returns:
            report: master report
        Example:
            >>> report = self._scrape_master_report(12345)
        """
        report = self._scrape_report(report_id)
        if report and report.dupe_of:
            return self._scrape_master_report(report.dupe_of)
        return report

    
    def _scrape_comments(self, report_id: int) -> list:
        """
        Scrape Bugzilla for comments.
        Args:
            report_id: id of the report to scrape.
        Returns:
            comments: list of comments (dict)
        Example:
            >>> comments = self._scrape_comments(12345)
        """
        try:
            scraped_comments = self.bugzilla.get_comments(report_id)
            return scraped_comments['bugs'][str(report_id)]['comments']
        except Exception as e:
            return None

    def _get_scraper_config(self, row: list) -> list:
        """
        Get the scraper configuration.
        Args:
            row: row from the scraper config csv.
        Returns:
            terms: list with scraper configuration.
        Example:
            >>> terms = self._get_scraper_config('scraper_config.csv')
        """
        terms = [
            {'limit': row[0]},
            {'product': row[1]}
        ]

        if row[2] == 'master':
            terms.append({'resolution': 'FIXED'})
            terms.append({'dupe_of': 'None'})
        elif row[2] == 'duplicate':
            terms.append({'resolution': 'DUPLICATE'})
            
        creation_time = datetime.strptime(row[3], '%Y-%m-%d')
        creation_time = creation_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        terms.append({'creation_time': creation_time})
        
        return terms

    def scrape(self, scraper_config_path: str) -> dict:
        """
        Scrape Bugzilla for reports.
        Args:
            scraper_config_path: path to the scraper config csv.
        Returns:
            dict: {'master': int, 'duplicate': int} 
            Number of master and duplicate reports scraped.
        """
        if not os.path.isfile(scraper_config_path):
            print("Invalid path") # TODO: change to exception
            sys.exit(1)
            
        result = {'master': 0, 'duplicate': 0} # Global result (all configs)

        path_no_extension = scraper_config_path.split('.')[0]
        results_path = f'{path_no_extension}_results.csv'
        with open(results_path, 'w') as results_file:
            # Set column names
            writer = csv.writer(results_file)
            writer.writerow([
                'num_reports', 'product', 'mode', 
                'creation_time', 'master_inserted', 'duplicate_inserted'
            ])

        with open(scraper_config_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Each config (row) results
                row_master_created = 0
                row_duplicate_created = 0

                terms = self._get_scraper_config(row)
                print(f"[*] Scraping for {terms}...")
                reports = self._scrape_reports(terms)
                num_created = create_many_reports(reports)
                print(f"[+] Found {len(reports)} reports. Created {num_created} reports")
                
                if row[2] == 'master':
                    row_master_created = num_created
                    result['master'] += num_created
                elif row[2] == 'duplicate':
                    row_duplicate_created = num_created
                    result['duplicate'] += num_created

                    # Search for the duplicate report
                    print(f"[*] Searching for master reports for duplicate ones...")
                    master_reports = []
                    for report in reports:
                        report_idx = reports.index(report) + 1
                        if report_idx % (len(reports) // 10) == 0:
                            print(f"[*] {report_idx}/{len(reports)}")

                        if report.dupe_of: # Should have one
                            master_report = self._scrape_master_report(report.dupe_of)
                            if master_report:
                                master_reports.append(master_report)
                            else:
                                print(f"[!] Could not get master report {report.dupe_of}. Deleting duplicated report {report.report_id}")
                                delete_report(report.report_id)
                                row_duplicate_created -= 1
                                result['duplicate'] -= 1
                        else:
                            print(f"[!] Report {report.report_id} does not have a dupe_of. Deleting...")
                            delete_report(report.report_id)
                            row_duplicate_created -= 1
                            result['duplicate'] -= 1
                    
                    row_master_created = create_many_reports(master_reports)
                    print(f"[+] Found {len(master_reports)} master reports. Created {row_master_created} reports")
                    result['master'] += row_master_created

                print("\n")
                with open(results_path, 'a') as resultscsvfile:
                    writer = csv.writer(resultscsvfile)
                    writer.writerow([row[0], row[1], row[2], row[3], max(0,row_master_created), max(0,row_duplicate_created)])

        result['master'] = max(0, result['master'])
        result['duplicate'] = max(0, result['duplicate'])
        return result
    
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 scraper.py <path_to_scrape_conf>")
        sys.exit(1)

    print("Scraping Bugzilla for reports...")
    path = sys.argv[1]
    print(f"[+] Scraper config: {path}\n")
    scraper = Scraper()
    result = scraper.scrape(path)
    print("\n[+] FINISHED")
    print(f"[+] Scraped {result['master']} master reports and {result['duplicate']} duplicate reports.")
    print(f"[+] Total reports: {result['master'] + result['duplicate']}")
