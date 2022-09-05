"""Module for SIDRD funcionality."""
import random
from models import TokenizedReport
from string import punctuation

import pickle
import pandas as pd
import numpy as np

import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


RESOURCES_PATH = 'resources'
TOKENIZER_PATH = 'resources/tokenizer'
VECTORIZER_PATH = 'resources/vectorizer'
CLUSTERIZER_PATH = 'resources/clusterizer'
CLASSIFIER_PATH = 'resources/classifier'

def dump_obj(obj, filename):
    """Persist an object to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    """Load an object from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)



class Tokenizer():

    def __init__(self, default_mode: str=None) -> None:
        self.__DEFAULT_MODE = default_mode if default_mode else 'stem'
        self.__EXTRA_CHARACTERS = [p for p in punctuation if p not in list('_')]
        self.__URL_FORBIDDEN_CHARS = [c for c in self.__EXTRA_CHARACTERS if c in [':', '/', '?', '=', '&', '#', '.']]
        self.__CUSTOM_WORDS = ['info', 'https', 'http', 'org', 'com', 'net', 'edu']
        self.__CUSTOM_WORDS.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])

        self.__LEMMATIZER = WordNetLemmatizer()
        self.__LEMMATIZER_PASS_TOKENS = ['js']
        self.__STEMMER = SnowballStemmer('english')
        self.__STEMMER_PASS_TOKENS = ['js']

    def __process_token(self, token: str) -> list:
        """
        Process a single token. Removes digits, extra characters and lowercases the token.
        Args:
            token: Token to process.
        Returns:
            List of processed tokens. (str)
        """
        token = token.lower()

        if token.startswith('http'):

            token = ''.join(
                [w if w not in self.__URL_FORBIDDEN_CHARS else ' ' for w in token]
            )
        else:
            token = ''.join(
                [w if w not in self.__EXTRA_CHARACTERS else ' ' for w in token]
            )

        token = token.strip()
        tokens = token.split() if len(token.split()) > 1 else [token]
        tokens = [t for t in tokens if not t.isdigit()]
        return tokens

    def __remove_extra_characters(self, text: str) -> list:
        """Remove extra characters from a sentence.
        Removes digits, extra characters and lowercases the token.
        Args:
            text (str): Text to process.
        Returns:
            list: list of processed tokens. (str)
        """
        sentence_to_process = text.split() # Generar tokens por espacios
        sentence_processed = []

        # Eliminar tokens que sean solo números
        sentence_processed = [w for w in sentence_to_process if not w.isdigit()]
        # Eliminar tokens que sean solo puntuación
        sentence_processed = [w for w in sentence_processed if w not in list(punctuation)]

        result = []
        for w in sentence_processed:
            token = self.__process_token(w)
            if len(token) > 1:
                for subtoken in token:
                    result.extend(self.__process_token(subtoken))
            else:
                result.extend(token)
        
        return result
    

    def __remove_stopwords(self, tokens: list) -> list:
        """Remove stop words from the lsit of tokens

        Args:
            tokens (list): each token is a string

        Returns:
            list: _description_
        """
        return [
            w for w in tokens if w not in stopwords.words('english') and w not in self.__CUSTOM_WORDS
            ]


    def __lemmatize(self, tokens: list) -> list:
        return [
            self.__LEMMATIZER.lemmatize(w) if w not in self.__LEMMATIZER_PASS_TOKENS else w for w in tokens
        ]


    def __stem(self, tokens: list) -> list:
        return [
            self.__STEMMER.stem(w) if w not in self.__STEMMER_PASS_TOKENS else w for w in tokens
        ]
    
    def generate_text(self, report: TokenizedReport, mode: str) -> str:
        """
        Generates text using the information from the report
        Args:
            report (TokenizedReport): Report to use.
            mode (str): Mode to use:
                S - summary
                SC - summary and component
                SCH - summary, component and bodies with High number of characters
                SCL - summary, component and bodies with Low number of characters
        Exceptions:
            ValueError: If mode is not valid or no information is available.
        Returns:
            str: Text generated from the report.
        """
        information = []
        if 'S' in mode:
            information.append(report.summary)
        if 'C' in mode:
            information.append(report.component)
        if 'H' in mode:
            information.append(report.comments) if len(report.comments) <= 1000 else None
        elif 'L' in mode:
            information.append(report.comments) if len(report.comments) <= 250 else None
 
        if len(information) == 0:
            raise ValueError('Invalid mode')
        
        # Double join with split to remove possible multiple spaces
        return ' '.join(' '.join(information).split())


    def tokenize(self, text: str, mode: str = None) -> list:
        """
        Processes a sentence. 
        Lowercases, removes digits and punctuation, stopwords and gets the root token.
        Args:
            text (str): Text to process.
            mode (str): Mode to process the text. Can be 'stem' or 'lemmatize'. Defaults to 'stem'.
        Returns:
            list: list of processed tokens. (str)
        """
        p_text = text.lower()
        p_text = self.__remove_extra_characters(p_text)
        p_text = self.__remove_stopwords(p_text)
        p_mode = mode if mode else self.__DEFAULT_MODE
        if p_mode == 'stem':
            p_text = self.__stem(p_text)
        elif p_mode == 'lemmatize':
            p_text = self.__lemmatize(p_text)
        return [t for t in p_text if t not in ['', ' ']]
    
    def retrain(self, new_config: dict):
        """
        Updates the tokenizer with the new config.
        Args:
            new_config (dict): New config to use.
        """
        self.__DEFAULT_MODE = new_config['default_mode']
        self.__EXTRA_CHARACTERS = new_config['extra_characters']
        self.__URL_FORBIDDEN_CHARS = new_config['url_forbidden_chars']
        self.__CUSTOM_WORDS = new_config['custom_words']
        self.__LEMMATIZER_PASS_TOKENS = new_config['lemmatizer_pass_tokens']
        self.__STEMMER_PASS_TOKENS = new_config['stemmer_pass_tokens']


class Vectorizer():

    def __init__(self, vectorizer=None) -> None:
        """
        If a vectorizer is not provided, the default one is used
        """
        self.__VECTORIZER = vectorizer if vectorizer else load_obj(
            f"{VECTORIZER_PATH}/tfidf-default.pkl")

    def vectorize(self, tokens: list) -> list:
        """
        Vectorizes a sentence or a list of sentences
        Args:
            tokens (list): list of lists of tokens to process.
                If single report, it is a list with one element.
                So, each element of the list is a list of tokens.
        Returns:
            list: list of Tf-idf-weighted document-term matrix.
        """
        x = []
        for t in tokens:
            x.append(' '.join(t))
        return self.__VECTORIZER.transform(x).toarray()

    def retrain(self, tokens_lists: list) -> list:
        """
        Fits a vectorizer with tokens_lists. Persists the vectorizer in a pickle file.
        Args:
            tokens_lists (list): List of lists of tokens. (each token list is a report)
        Returns:
            list: list of Tf-idf-weighted document-term matrix in a numpy array
        """
        x = [' '.join(tokens) for tokens in tokens_lists]
        self.__VECTORIZER = TfidfVectorizer()
        features = self.__VECTORIZER.fit_transform(x)
        dump_obj(self.__VECTORIZER, f'{VECTORIZER_PATH}/tfidf.pkl')
        return features


class Clusterizer():

    __REP_CLU = 'REPORTS_PER_CLUSTER'
    __REC_LVL = 'RECURSIVITY_LEVEL'

    def __init__(self, vectorizer: Vectorizer = None, n_clusters: int = 2, 
            limit: int = 5, mode: str = __REC_LVL, min_reports_per_cluster = 3
        ):
        self.__VECTORIZER = vectorizer if vectorizer else load_obj(
            f'{RESOURCES_PATH}/vectorizer-default.pkl')
        self.__N_CLUSTERS = n_clusters
        # limit can be iteration limit or number of reports in final cluster
        self.__LIMIT = limit 
        self.__MODE = mode
        self.__MIN_REPORTS = min_reports_per_cluster
    
    def __build_reports_dataframe(self, reports: list, report: TokenizedReport) -> pd.DataFrame:
        """
        Builds a dataframe with the reports data.
        Adds in the end the new report
        Args:
            reports (list of TokenizedReport): List of reports.
            report (TokenizedReport): New report to add.
        Returns:
            pd.DataFrame: Dataframe with the reports data.
        """
        # Create base dataframe
        columns = ['report_id', 'dupe_of', 'creation_time', 'status', 'component', 'summary', 'comments', 'text', 'tokens']
        df = pd.DataFrame(columns=columns)
        # Add DB reports
        for rep in reports:
            rep_df = pd.DataFrame([[
                rep.report_id, rep.dupe_of, rep.creation_time, rep.status, rep.component,
                rep.summary, rep.comments, rep.text, rep.tokens
            ]], columns=columns)
            df = pd.concat([df, rep_df])
        # Add analyzed report
        rep_df = pd.DataFrame([[
            report.report_id, report.dupe_of, report.creation_time, report.status,
            report.component, report.summary, report.comments,
            report.text, report.tokens
        ]], columns=columns)
        df = pd.concat([df, rep_df])
        return df

    def clusterize(self, report: TokenizedReport, reports_db: list) -> pd.DataFrame:
        """
        Receives a list of features representing a new report.
        Obtains the tokens of the reports in DB, transforms them into features.
        Clusters the features iteratively.
        Args:
            report (TokenizedReport): The new report to clusterize. Must have 'tokens' and 'report_id'
            reports_db (list): list of reports to clusterize. TokenizedReport elements.
        Returns:
            pd.DataFrame: list of reports in the same cluster as the new report
        """
        # Build dataframe
        df = self.__build_reports_dataframe(reports_db, report)

        limit = 0 if self.__MODE == self.__REC_LVL else self.__LIMIT
        current = self.__LIMIT if self.__MODE == self.__REC_LVL else len(df)

        while all([current > limit, len(df) >= self.__N_CLUSTERS]):
            # Transform tokens into features:
            features = self.__VECTORIZER.vectorize(df['tokens'])
            # Fit cluster model with features
            kmeans = KMeans(n_clusters=self.__N_CLUSTERS, random_state=0).fit(features)
            # Keep last cluster in case new one is not big enough
            last_df = df.copy()
            # Assign the labels to the dataframe
            df['cluster'] = kmeans.labels_
            # Get the cluster of the new report
            new_report_cluster = df[df['report_id'] == report.report_id]['cluster'].values[0]
            # Filter the dataframe to get the reports in the same cluster as the new report
            df = df[df['cluster'] == new_report_cluster]
            # Update current
            current = current-1 if self.__MODE == self.__REC_LVL else len(df)

        # If the cluster is too small, return the last cluster
        df = df if len(df) >= self.__MIN_REPORTS else last_df

        # Remove report from similar reports dataframe
        df = df[df['report_id'] != report.report_id]
        return df

    def retrain(self, vectorizer: Vectorizer, new_config: dict) -> None:
        """
        Retrains the clusterizer with new parameters.
        Args:
            vectorizer (Vectorizer): Vectorizer to use.
            new_config (dict): Dictionary with the new parameters.
        """
        self.__VECTORIZER = vectorizer
        self.__N_CLUSTERS = new_config['n_clusters']
        self.__LIMIT = new_config['limit']
        self.__MODE = new_config['mode']
        self.__MIN_REPORTS = new_config['min_reports_per_cluster']


class Classifier():
    
    def __init__(self, classifier=None, vectorizer=None) -> None:
        """
        If no classifier is provided, it loads the default classifier.
        If no vectorizer is provided, it loads the default vectorizer.
        """
        self.__CLASSIFIER = classifier if classifier else load_obj(
            f"{CLASSIFIER_PATH}/rfc-default.pkl")
        self.__VECTORIZER = vectorizer if vectorizer else load_obj(
            f"{RESOURCES_PATH}/vectorizer-default.pkl")

    def __build_dataframe(self, report: TokenizedReport, similar_reports: pd.DataFrame) -> pd.DataFrame:
        """
        Builds the dataframe containing the pairs <report>-<similar_reports_elements>
        Args:
            report (TokenizedReport): the report to analyze
            similar_reports (pd.DataFrame): list of similar reports, obtained after clustering
        Returns:
            pd.DataFrame: list of pairs.
        """
        # Create base dataframe
        columns = [
            'analyzed_report_id', 'similar_report_id', 'similar_report_summary',
            'similar_report_component', 'tokens']
        df = pd.DataFrame(columns=columns)
        # Create pairs
        for i, row in similar_reports.iterrows():
            report_id = row['report_id']
            summary = row['summary']
            component = row['component']
            tokens = row['tokens'] + report.tokens
            pair_df = pd.DataFrame([[
                report.report_id, report_id, summary, component, tokens
            ]], columns=columns)
            df = pd.concat([df, pair_df])
        return df
    
    def __build_retrain_dataframe(self, report_set: list, per_duplicate_pairs: float, verbose: bool = True) -> pd.DataFrame:
        """
        Builds the dataframe containing the pairs <report>-<similar_reports_elements>
        Args:
            report_set (list): list of TokenizedReport elements
            per_duplicate_pairs (float): percentage of duplicate pairs to include in the dataframe
            verbose (bool): if True, prints the progress
        Returns:
            pd.DataFrame: list of pairs.
        """
        n_reports = len(report_set)
        n_duplicates = int(n_reports * per_duplicate_pairs)
        n_non_duplicates = n_reports - n_duplicates

        duplicates = [report for report in report_set if str(report.dupe_of) != 'nan']
        masters = [report for report in report_set if str(report.dupe_of) == 'nan']

        # Create base dataframe
        columns = ["tokens", "duplicate"]
        df = pd.DataFrame(columns=columns)

        if verbose:
            print(f"\t[+] Number of duplicate pairs to create: {n_duplicates}")
            print(f"\t[+] Number of non-duplicate pairs to create: {n_non_duplicates}")
            print(f"\t[+] Number of available reports: {n_reports} ({len(duplicates)} duplicates, {len(masters)} masters)")
            print("\t[*] Creating duplicate pairs...")

        # Add duplicate pairs
        for report in duplicates:
            try:
                master = [master for master in masters if master.report_id == report.dupe_of][0]
                tokens = report.tokens + master.tokens
                df = pd.concat([df, pd.DataFrame([[tokens, 1]], columns=columns)])
            except IndexError:
                pass
            
            if len(df) >= n_duplicates:
                break
        if verbose:
            print(f"\t[+] Created {len(df)} duplicate pairs")
            print("\t[*] Creating non-duplicate pairs...")

        # Add master pairs
        added = 0
        while added < n_non_duplicates:
            master1 = random.choice(masters)
            master2 = random.choice(masters)
            if master1.report_id != master2.report_id:
                tokens = master1.tokens + master2.tokens
                if tokens not in df['tokens'].values:
                    df = pd.concat([df, pd.DataFrame([[tokens, 0]], columns=columns)])
                    added += 1
        
        df = df.sample(frac=1).reset_index(drop=True)

        if verbose:
            print(f"\t[+] Created {added} non-duplicate pairs")
            print("\t[*] Shuffling dataframe...")
            print(f"\t[+] Done. Created {len(df)} pairs")

        return df

    def get_possible_duplicates(self, report: TokenizedReport, similar_reports: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a tokenized report and a dataframe containing the similar reports.
        Builds a dataframe with the possible pairs.
        Classify each pair and returns the ones that are two duplicate reports.
        Args:
            report (TokenizedReport): The new report to clusterize. Must have 'tokens' and 'report_id'
            similar_reports (pd.DataFrame): list of reports in the same cluster as the new report
        Return:
            pd.DataFrame: list of reports that are duplicates of the new report
        """
        # Build pairs dataframe
        df = self.__build_dataframe(report, similar_reports)
        # Transform tokens into features
        features = self.__VECTORIZER.vectorize(df['tokens'])
        # Predict
        df['duplicate'] = self.__CLASSIFIER.predict(features)
        # Filter the dataframe to get the reports that are duplicates
        df = df[df['duplicate'] == 1]
        # Return reports
        return similar_reports[similar_reports['report_id'].isin(df['similar_report_id'])]

    def retrain(self, vectorizer: Vectorizer, report_set: list, new_config: dict, verbose: bool=True) -> None:
        """
        Retrains the classifier with new vectorizer and features
        Args:
            vectorizer (Vectorizer): Vectorizer to use.
            report_set (list): list of reports to train the classifier.
            new_config (dict): Dictionary with the parameters to use.
            verbose (bool): If True, adds verbosity to the process.
        """
        if verbose:
            print("\t[*] Building pairs dataframe...")
        per_duplicate_pairs = new_config['per_duplicate_pairs']
        df = self.__build_retrain_dataframe(report_set, per_duplicate_pairs)
        if verbose:
            print("\t[+] Done.")
        
        if verbose:
            print("\t[*] Vectorizing tokens...")
        self.__VECTORIZER = vectorizer
        features = self.__VECTORIZER.vectorize(df['tokens'])
        if verbose:
            print("\t[+] Done.")

        param_grid = new_config['param_grid']
        rfc = RandomForestClassifier()
        if verbose:
            print(f"\t[+] Searching for best parameters with grid: {param_grid}")

        verbosity = 1 if verbose else 0
        gs = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1, refit=True, verbose=verbosity)
        if verbose:
            print("\t[*] GridSearchCV process started.")
        gs.fit(features, df['duplicate'].astype('int'))

        self.__CLASSIFIER = gs.best_estimator_
        dump_obj(self.__CLASSIFIER, f"{CLASSIFIER_PATH}/rfc.pkl")
        if verbose:
            print("\t[+] GridSearchCV process finished.")
            print(f"\t[+] Best parameters: {gs.best_params_}")


class SIDRD():

    __MAX_REPORTS_TO_SHOW = 10

    def __init__(self, default_model: bool = True,
                tokenizer: Tokenizer = None, vectorizer: Vectorizer = None,
                clusterizer: Clusterizer = None, classifier: Classifier = None):
        """
        If default_model, uses the default model. Otherwise, uses the latest trained ones
        """
        if default_model:
            self.tokenizer = tokenizer if tokenizer else load_obj(f"{RESOURCES_PATH}/tokenizer-default.pkl")
            self.vectorizer = vectorizer if vectorizer else load_obj(f'{RESOURCES_PATH}/vectorizer-default.pkl')
            self.clusterizer = clusterizer if clusterizer else load_obj(f'{RESOURCES_PATH}/clusterizer-default.pkl')
            self.classifier = classifier if classifier else load_obj(f'{RESOURCES_PATH}/classifier-default.pkl')
        else:
            self.tokenizer = tokenizer if tokenizer else load_obj(f'{RESOURCES_PATH}/tokenizer.pkl')
            self.vectorizer = vectorizer if vectorizer else load_obj(f'{RESOURCES_PATH}/vectorizer.pkl')
            self.clusterizer = clusterizer if clusterizer else load_obj(f'{RESOURCES_PATH}/clusterizer.pkl')
            self.classifier = classifier if classifier else load_obj(f'{RESOURCES_PATH}/classifier.pkl')

    
    def __dataframe_to_dictionary_reports(self, df: pd.DataFrame) -> list:
        """
        Searches in BD for the reports and returns the list of dictionaries with keys 'report_id', 'summary', 'component' and 'description'
        Args:
            df (pd.DataFrame): dataframe to convert
        Returns:
            list: list of dicts
        """
        return [{
            'report_id': row['report_id'],
            'dupe_of': row['dupe_of'],
            'creation_time': row['creation_time'],
            'component': row['component'],
            'summary': row['summary'],
            'description': row['comments']
        } for i, row in df.iterrows()]

    def get_duplicates(self, report: TokenizedReport, reports_db: list) -> tuple:
        """
        Takes a tokenized report and returns the possible duplicates.
        Args:
            report (TokenizedReport): The new report to clusterize. Must have 'tokens' and 'report_id'
            reports_db (list): list of reports in the database. TokenizedReport objects
        Return:
            tuple: 
                - report tokenized
                - list of possible duplicates (dictionaries with keys 'report_id', 'summary', 'component', 'description')
        """
        text = self.tokenizer.generate_text(report, 'SC')
        report.text = text
        report.tokens = self.tokenizer.tokenize(text)
        similar_reports = self.clusterizer.clusterize(report, reports_db)
        duplicates = self.classifier.get_possible_duplicates(report, similar_reports)
        duplicates = duplicates.sort_values(by=['creation_time'], ascending=False)
        duplicates = duplicates[:self.__MAX_REPORTS_TO_SHOW] if self.__MAX_REPORTS_TO_SHOW < len(duplicates) else duplicates
        return report, self.__dataframe_to_dictionary_reports(duplicates)
    
    def retrain(self, reports_set: list, new_config: dict, verbose: bool=True) -> None:
        """
        Retrains the SIDRD components with the new reports.
        Args:
            reports_set (list): list of reports to retrain the model
            new_config (dict): dictionary with the new configuration
            verbose (bool): if True, prints the progress
        """
        if verbose:
            print('[*] Reconfiguring tokenizer...')
        # Tokenizer component loads new config
        self.tokenizer.retrain(new_config['tokenizer'])
        # Use the new values to generate the new tokens
        tokens = [self.tokenizer.tokenize(report.text) for report in reports_set]
        # Persist new tokenizer
        dump_obj(self.tokenizer, f"{RESOURCES_PATH}/tokenizer.pkl")
        if verbose:
            print(f'[+] Done. Stored in {RESOURCES_PATH}/tokenizer.pkl')

        if verbose:
            print('[*] Retraining vectorizer...')
        # Train another vectorizer with the new tokens and get the new features
        features = self.vectorizer.retrain(tokens)
        # Persist new vectorizer
        dump_obj(self.vectorizer, f"{RESOURCES_PATH}/vectorizer.pkl")
        if verbose:
            print(f'[+] Done. Stored in {RESOURCES_PATH}/vectorizer.pkl')

        if verbose:
            print('[*] Reconfiguring clusterizer...')
        # Clusterizer component loads the new config
        self.clusterizer.retrain(self.vectorizer, new_config['clusterizer'])
        # Persist new clusterizer
        dump_obj(self.clusterizer, f"{RESOURCES_PATH}/clusterizer.pkl")
        if verbose:
            print(f'[+] Done. Stored in {RESOURCES_PATH}/clusterizer.pkl')

        if verbose:
            print('[*] Retraining classifier...')
        # Train another classifier 
        self.classifier.retrain(self.vectorizer, reports_set, new_config['classifier'], verbose)
        # Persist new classifier
        dump_obj(self.classifier, f"{RESOURCES_PATH}/classifier.pkl")
        if verbose:
            print(f'[+] Done. Stored in {RESOURCES_PATH}/classifier.pkl')
