"""Main module while no api routes"""
import os
import sys
import getopt

from controllers import (
    cli_get_possible_duplicates, cli_create_report, 
    get_number_of_reports,
    retrain_sidrd
)
from scraper.scraper import Scraper

import warnings
warnings.filterwarnings('ignore')

def print_help():
    help_msg = "\nUsage: python3 main.py [-h] [--help]\n" \
    "\t[-s config_file]\n" \
    "\t[-c]\n" \
    "\t[-r config_file]\n" \
    "---------------------------------------------------------------------------------------------------------\n" \
    "-h, --help\t\tPrint this help message.\n" \
    "-s <config_file>\tScrape and persist reports according config file.\n" \
    "\t\t\t\tMust be inside scraper/config/ and have the format of scraper_config.csv.example.\n" \
    "\t\t\t\tMore information in README.md\n" \
    "\t\t\t\tExample: python3 main.py -s scraper_config.csv\n" \
    "-c\t\t\tCreate a new report using SIDRD to get possible duplicates\n" \
    "\t\t\t\tExample: python3 main.py -c\n" \
    "-r <config_path>\tRetrain SIDRD\n" \
    "\t\t\t\tExample: python3 main.py -r resources/sidrd_retrain_config.json\n"
    print(help_msg)

def print_report(rep):
    report = "---------------------------------------------------------------------------------------------------------\n" \
            f"""Report: {rep['report_id']} ({f"Duplicado de {rep['dupe_of']}" if rep['dupe_of'] else "Reporte maestro"}) | Component: {rep['component']}\n""" \
            f"Date: {rep['creation_time']}\n" \
            "---------------------------------------------------------------------------------------------------------\n" \
            f"\nSummary:\n{rep['summary']}\n" \
            f"\nDescription:\n{rep['description']}\n" \
            "---------------------------------------------------------------------------------------------------------\n\n"
    print(report)
    
            

if __name__ == "__main__":
    
    # Get arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:cr:", ["help"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o in ("-h", "--help"):
            print_help()
            sys.exit(0)
        elif o == "-s":
            print("[+]" + "-"*25 + " SCRAPING MODE SELECTED " + "-"*25)
            database = os.environ.get('DB_NAME')
            collection = os.environ.get('DB_REPORT_COLLECTION')
            print(f"[+] Database: {database}. Collection: {collection}")
            result = Scraper().scrape(a)
            print("[+]" + "-"*25 + "    SCRAPING FINISHED   " + "-"*25)
            sys.exit(0)
        elif o == "-c":
            print("[+]" + "-"*25 + "  CREATE REPORT MODE SELECTED  " + "-"*25)
            database = os.environ.get('DB_NAME')
            collection = os.environ.get('DB_TOKENIZED_COLLECTION')
            n_reports = get_number_of_reports()
            print(f"[+] Database: {database}. Collection: {collection} ({n_reports} reports)")
            print(f"[!] If this is not the collection you want to create reports from, please abort using Ctrl+C.")
            input(f"[!] If this is the correct collection, press ENTER...")
            print("[+] Introduce the following data:")
            component = input("[+] Component: ")
            summary = input("[+] Summary: ")
            description = input("[+] Description: ")
            print("[+] Would you like to use the default model or the last trained model?")
            opt = int(input("[+] Default model (0: no, 1: yes): "))
            default_model = False if opt == 0 else True
            print(f"[*] Using SIDRD to get possible duplicates...")
            report, possible_duplicates = cli_get_possible_duplicates(component, summary, description, default_model)
            print(f"[+] Done.")
            if len(possible_duplicates) == 0:
                print(f"[!] No possible duplicates found.")
                print(f"[!] If you still want to create the report press ENTER. Otherwise abort using Ctrl+C.")
                input()
            else:
                input(f"[!] FOUND {len(possible_duplicates)} possible duplicates. Press ENTER to show")
                for duplicate in possible_duplicates[::-1]:
                    print_report(duplicate)
                print(f"[!] If any of this reports correspond to the one you want to create, please type its ID and press ENTER.")
                print(f"[!] If none of this reports correspond to the one you want to create, please type 0 press ENTER.")
                print(f"[!] If you want to abort, please press Ctrl+C.")
                dupe_of = int(input(">>> "))
                print(f"[*] Creating report in DB...")
                cli_create_report(report, dupe_of)
                print(f"[+] Done.")
            print(f"[+]" + "-"*25 + "  CREATE REPORT FINISHED  " + "-"*25)
            sys.exit(0)
        elif o == "-r":
            print("[+]" + "-"*25 + "  RETRAIN MODE SELECTED  " + "-"*25)
            database = os.environ.get('DB_NAME')
            collection = os.environ.get('DB_TOKENIZED_COLLECTION')
            n_reports = get_number_of_reports()
            print(f"[+] Database: {database}. Collection: {collection} ({n_reports} reports)")
            print(f"[+] Using config file {a}")
            print(f"[!] If this is not the collection or configuration you want to retrain with, please abort using Ctrl+C.")
            input(f"[!] If this is the correct collection, press ENTER...")
            verbose = input(f"[+] Use verbosity? (0: no, 1: yes): ")
            verbose = False if verbose == "0" else True
            retrain_sidrd(a, verbose)
            print(f"[+]" + "-"*25 + "  RETRAIN FINISHED  " + "-"*25)

