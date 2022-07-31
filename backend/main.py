"""Main module while no api routes"""

import sys
import getopt

from scraper.scraper import Scraper

def print_help():
    help_msg = "Usage: python3 main.py [-h] [--help]\n" \
    "\t[-s config_file]\n" \
    "---------------------------------------------------------------------------------------------------------\n" \
    "-h, --help\t\tPrint this help message.\n" \
    "-s <config_file>\tScrape and persist reports according config file.\n" \
    "\t\t\t\tMust be inside scraper/config/ and have the format of scraper_config.csv.example.\n" \
    "\t\t\t\tMore information in README.md\n" \
    "\t\t\t\tExample: python3 main.py -s scraper_config.csv"
    print(help_msg)

if __name__ == "__main__":
    
    # Get arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:", ["help"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for o, a in opts:
        if o in ("-h", "--help"):
            print_help()
            sys.exit(0)
        elif o == "-s":
            print("[+]" + "-"*25 + " SCRAPING MODE SELECTED " + "-"*25)
            result = Scraper().scrape(a)
            print("[+]" + "-"*25 + "    SCRAPING FINISHED   " + "-"*25)
            sys.exit(0)
