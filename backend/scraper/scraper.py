"""Module for scraping interface"""
import os
import sys
import csv
from datetime import datetime

from controllers import create_many_reports, delete_report
from models import Report
from scraper.bugzilla_scraper import BugzillaBaseScraper

SCRAPER_CONFIG_DIR = "scraper/config/"

class Scraper():
    """
    Class for scraping interface.
    """

    def __init__(self):
        self.bugzilla = BugzillaBaseScraper()
 
    def __parse_report(self, scraped_report: dict) -> Report:
        """
        Parse a report from the Bugzilla API.
        Args:
            scraped_report: report from the Bugzilla API.
        Returns:
            report:Report report from the database.
        Example:
            >>> report = self.__parse_report(scraped_report)
        """
        report_id = int(scraped_report['id'])
        creation_time = datetime.strptime(scraped_report['creation_time'], 
                                        '%Y-%m-%dT%H:%M:%SZ')
        status = scraped_report['status']
        component = scraped_report['component']
        dupe_of = int(scraped_report['dupe_of']) if scraped_report['dupe_of'] else None
        summary = scraped_report['summary']
        try:
            comments = scraped_report['comments'][f'{report_id}']['comments'][0]['raw_text']
        except Exception as e:
            comments = ""

        return Report(report_id, creation_time, status, 
                    component, dupe_of, summary, comments)
                    

    def __scrape_reports(self, terms: dict) -> list:
        """
        Scrape Bugzilla for reports.
        Args:
            terms: dict with terms to search for.
        Returns:
            reports: list of reports (Report)
        """
        try:
            scraped_reports = self.bugzilla.search_bugs(terms)
            parsed_reports = [self.__parse_report(scraped_report) for scraped_report in scraped_reports['bugs']]
            return parsed_reports
        except Exception as e:
            return None

    def __scrape_report(self, report_id: int) -> Report:
        """
        Scrape Bugzilla for a single report.
        Args:
            report_id: id of the report to scrape.
        Returns:
            report:Report report from the database.
        Example:
            >>> report = self.__scrape_report(12345)
        """
        try:
            scraped_report = self.bugzilla.search_bug(report_id)
            return self.__parse_report(scraped_report)
        except Exception as e:
            return None
    
    def __scrape_master_report(self, report_id: int) -> Report:
        """
        Scrape Bugzilla for the master of a given report.
        If the retrieved report is duplicated, returns the corresponding master.
        Args:
            report_id: id of the report to scrape.
        Returns:
            report: master report
        Example:
            >>> report = self.__scrape_master_report(12345)
        """
        report = self.__scrape_report(report_id)
        if report and report.dupe_of:
            return self.__scrape_master_report(report.dupe_of)
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

    def __get_scraper_config(self, row: list) -> dict:
        """
        Get the scraper configuration.
        Args:
            row: row from the scraper config csv.
        Returns:
            terms: dict with scraper configuration.
        Example:
            >>> terms = self.__get_scraper_config('scraper_config.csv')
        """
        terms = {
            'limit': row[0],
            'product': row[1],
            'creation_time': row[3]
        }

        if row[2] == 'master':
            terms['resolution'] = 'FIXED'
            terms['dupe_of'] = 'None'
        elif row[2] == 'duplicate':
            terms['resolution'] = 'DUPLICATE'

        return terms
    
    def __get_expected_reports(self, configs: list) -> dict:
        """
        Gets the expected number of reports to scrape based on the configurations
        Args:
            configs: list of scraper configurations.
        Returns:
            expected_reports: dict with the expected reports.
                Example: {'master': 100, 'duplicate': 50}
        Example:
            >>> expected_reports = self.__get_expected_reports(configs)
        """
        expected_reports = {'master': 0, 'duplicate': 0}
        for c in configs:
            if c['resolution'] == 'DUPLICATE':
                expected_reports['duplicate'] += int(c['limit'])
                expected_reports['master'] += int(c['limit'])
            else:
                expected_reports['master'] += int(c['limit'])
        return expected_reports

    def scrape(self, scraper_config_file: str) -> dict:
        """
        Scrape Bugzilla for reports.
        Writes results in scraper/config/<scraper_config_file>_results.csv.
        Args:
            scraper_config_path: path to the scraper config csv.
                Must be inside scraper/config/.
                Must have the right format (see scraper/config/scraper_config.csv.example).
        Returns:
            dict: {'master': int, 'duplicate': int} 
            Number of master and duplicate reports scraped.
        Example:
            >>> scraper = Scraper()
            >>> results = scraper.scrape('scraper_config.csv.example')
        """
        time_start = datetime.now()
        # Check if the scraper config file exists in scraper/config folder
        print("[*] Checking scraper config file...")
        try:
            if not os.path.isfile(SCRAPER_CONFIG_DIR + scraper_config_file):
                print('[!] Scraper config file does not exist. Exiting...')
                sys.exit(1)
        except Exception as e:
            print('[!] Error:', e)
            sys.exit(1)
        print(f"[+] Scraper config file OK: {SCRAPER_CONFIG_DIR + scraper_config_file}")

        # Initialize result
        result = {'master': 0, 'duplicate': 0} # Global result (all configs)

        # Initialize result file
        file_no_extension = scraper_config_file.split('.')[0]
        results_path = f'{SCRAPER_CONFIG_DIR+file_no_extension}_results.csv'
        with open(results_path, 'w') as results_file:
            # Set column names
            writer = csv.writer(results_file)
            writer.writerow([
                'num_reports', 'product', 'mode', 
                'creation_time', 'master_inserted', 'duplicate_inserted'
            ])

        # Read config file
        with open(SCRAPER_CONFIG_DIR+scraper_config_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            configs = [self.__get_scraper_config(row) for row in reader]

        num_configs = len(configs)
        expected_reports = self.__get_expected_reports(configs)
        total_expected = expected_reports['master'] + expected_reports['duplicate']

        print(f"[+] Starting scraper with {num_configs} configs. Expected {total_expected} reports ", end="")
        print(f"({expected_reports['master']} master, {expected_reports['duplicate']} duplicate)")

        # Each row is a different config (ex: 100 master reports from firefox)
        for terms in configs:
            # Initialize config results
            config_master_created = 0
            config_duplicate_created = 0
            print("[+]" + "-"*20 + f" Config ({configs.index(terms)+1}/{num_configs}) " + "-"*20)
            print(f"[*] Scraping config: {terms}\n[+] ...", end="")

            # Get reports from Bugzilla and save in BD
            reports = self.__scrape_reports(terms)
            num_created = create_many_reports(reports)
            print(f"Found {len(reports)} reports. Created {num_created} reports")

            # Depending on mode, update result
            if terms['resolution'] != 'DUPLICATE': # master
                config_master_created = num_created
                result['master'] += num_created
            else: # duplicate
                config_duplicate_created = num_created
                result['duplicate'] += num_created

                # Search for the master report of each duplicate report
                master_reports = []
                print(f"[*] Searching for master reports for duplicate ones...")

                for report in reports:
                    if report.dupe_of: # Should have one, if not, it is deleted
                        master_report = self.__scrape_master_report(report.dupe_of)
                        # If the master can not be found, corresponding duplicate is deleted
                        if master_report:
                            master_reports.append(master_report)
                        else:
                            print(f"[!] Could not get master report {report.dupe_of}. Deleting duplicated report {report.report_id}")
                            delete_report(report.report_id)
                            config_duplicate_created -= 1
                            result['duplicate'] -= 1
                    else:
                        print(f"[!] Report {report.report_id} does not have a dupe_of. Deleting...")
                        delete_report(report.report_id)
                        config_duplicate_created -= 1
                        result['duplicate'] -= 1

                    # Visual debug progress
                    report_idx = reports.index(report) + 1
                    if report_idx % (len(reports) // 5) == 0:
                        print(f"[*] {report_idx}/{len(reports)} duplicates processed")

                # After finding masters, save in BD and update results
                config_master_created = create_many_reports(master_reports)
                print(f"[+] Found {len(reports)} duplicate reports. Created {config_duplicate_created} duplicate reports")
                print(f"[+] Found {len(master_reports)} master reports. Created {config_master_created} master reports")
                result['master'] += config_master_created

            # Write config results to file
            with open(results_path, 'a') as resultscsvfile:
                writer = csv.writer(resultscsvfile)
                writer.writerow([
                    terms['limit'], terms['product'], 'duplicate' if terms['resolution'] == 'DUPLICATE' else 'master',
                    terms['creation_time'], max(0, config_master_created), max(0, config_duplicate_created)
                ])

        result['master'] = max(0, result['master'])
        result['duplicate'] = max(0, result['duplicate'])
        print("[+]" + "-"*18 + f" Scraping results " + "-"*18)
        print(f"[+] Scraped {result['master']} master reports and {result['duplicate']} duplicate reports.")
        print(f"[+] Total reports: {result['master'] + result['duplicate']} out of {total_expected} expected.")
        elapsed_time = datetime.now() - time_start
        print(f"[+] Scraper finished in {elapsed_time}")
        return result
