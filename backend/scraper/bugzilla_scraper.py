"""Base Scraper for Bugzilla"""
import os
import requests

BUGZILLA_BASE_URL = "https://bugzilla.mozilla.org/rest/"
BUGZILLA_API_KEY = os.environ.get('BUGZILLA_API_KEY', '')
AVAILABLE_TERMS = ['limit', 'product', 'resolution', 'dupe_of', 'creation_time']

class BugzillaBaseScraper():

    def __init__(self):
        pass

    def __get_request(self, url: str) -> dict:
        """
        Make a GET request to the Bugzilla API.
        Args:
            url: url to make the request to.
        Returns:
            response: response from the request.
        """
        headers = {'X-Bugzilla-API-Key': BUGZILLA_API_KEY}
        response = requests.get(url, headers=headers)
        return response.json()

    def __get_query_from_terms(self, terms: dict) -> dict:
        """
        Get the parameters query from the terms.
        Args:
            terms: terms to search for.
        Returns:
            query: parameters query to search for.
        """
        query = ""
        for k,v in terms.items():
            if k in AVAILABLE_TERMS:
                query += f"{k}={v}&"
        return query[:-1] if query else ""

    def search_bug_comments(self, bug_id: int) -> dict:
        """
        Search for comments on a bug.
        Args:
            bug_id: id of the bug to search for.
        Returns:
            comments: comments from the Bugzilla API.
        """
        url = f"{BUGZILLA_BASE_URL}bug/{bug_id}/comment"
        response = self.__get_request(url)
        comments = response['bugs']
        return comments

    def search_bug(self, bug_id: int) -> dict:
        """
        Search for a bug by id.
        Args:
            bug_id: id of the bug to search for.
        Returns:
            bug: bug from the Bugzilla API.
        """
        url = f"{BUGZILLA_BASE_URL}bug/{bug_id}"
        bug = self.__get_request(url)
        bug =  bug['bugs'][0] if 'bugs' in bug else bug
        bug['comments'] = self.search_bug_comments(bug_id)
        return bug

    def search_bugs(self, terms: dict) -> dict:
        """
        Search for bugs.
        Args:
            terms: dict with terms to search for (see bugzilla_scraper.AVAILABLE_TERMS)
        Returns:
            bugs: bugs from the Bugzilla API.
        """
        base_url = f"{BUGZILLA_BASE_URL}bug?"
        query = self.__get_query_from_terms(terms)
        url = f"{base_url}{query}"
        bugs = self.__get_request(url)
        for bug in bugs['bugs']:
            bug['comments'] = self.search_bug_comments(bug['id'])
        return bugs
