"""Module for SIDRD funcionality."""
from curses.ascii import isdigit
from typing import Optional
from controllers import (
    get_tokenized_reports as get_tokenized_reports_controller
)
from string import punctuation

import pickle
import pandas as pd

import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer


RESOURCES_PATH = 'resources'

def dump_obj(obj, filename):
    """Persist an object to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    """Load an object from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


class SIDRD():

    def __init__(self) -> None:

        self.tokenizer = Tokenizer()
        self.vectorizer = Vectorizer()
        self.clusterizer = Clusterizer()
        self.classifier = Classifier()

    
    def get_duplicates(self, report) -> None:
        # TODO: Cambiar en funcion de resultados
        text = report.summary + ' ' + report.component
        
        tokens = self.tokenizer.tokenize(text)
        
        features = self.vectorizer.vectorize(tokens)
        
        cluster_reports = self.clusterizer.clusterize(features)
        
        duplicates = self.classifier.classify(cluster_reports)
        
        return duplicates
    
    def retrain(self) -> None:
        pass


class Tokenizer():

    def __init__(self) -> None:
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


    def tokenize(self, text: str, mode: str) -> list:
        """
        Processes a sentence. 
        Lowercases, removes digits and punctuation, stopwords and gets the root token.
        Args:
            text (str): Text to process.
            mode (str): Mode to process the text. Can be 'stem' or 'lemmatize'.
        Returns:
            list: list of processed tokens. (str)
        """
        p_text = text.lower()
        p_text = self.__remove_extra_characters(p_text)
        p_text = self.__remove_stopwords(p_text)
        if mode == 'stem':
            p_text = self.__stem(p_text)
        elif mode == 'lemmatize':
            p_text = self.__lemmatize(p_text)
        return [t for t in p_text if t not in ['', ' ']]


class Vectorizer():

    def __init__(self, vectorizer=None) -> None:
        self.__vectorizer_path = f'{RESOURCES_PATH}/vectorizer.pkl'
        self.__VECTORIZER = vectorizer if vectorizer else load_obj(self.__vectorizer_path)
    
    def vectorize(self, tokens: list) -> list:
        """
        Vectorizes a sentence or a list of sentences
        Args:
            tokens (list): tokens to process. 
                If single report, it is a list with one element.
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
            list: list of Tf-idf-weighted document-term matrix.
        """
        x = [' '.join(tokens) for tokens in tokens_lists]
        self.__VECTORIZER = TfidfVectorizer()
        features = self.__VECTORIZER.fit_transform(x)
        dump_obj(self.__VECTORIZER, self.__vectorizer_path)
        return features

class Clusterizer():

    def __init__(self, vectorizer: Vectorizer) -> None:
        self.__N_CLUSTERS = 3
        self.__LIMIT = 100 # Can be iteration limit or number of reports in final cluster
        self.__VECTORIZER = vectorizer
    
    def __build_reports_dataframe(reports: list) -> pd.DataFrame:
        """
        Builds a dataframe with the reports data.
        Args:
            reports (list): List of reports.
        Returns:
            pd.DataFrame: Dataframe with the reports data.
        """
        columns = ['report_id', 'creation_time', 'status', 'component', 'summary', 'comments', 'text', 'tokens']
        df = pd.DataFrame(columns=columns)
        # Add report (TokenizedReport) in reports to df
        for report in reports:
            pass

    def clusterize(self, features: str, test_reports: Optional[list]=None) -> pd.DataFrame:
        """
        Receives a list of features representing a new report.
        Obtains the tokens of the reports in BD, transforms them into features.
        Clusters the features iteratively.
        Args:
            features (list): list of features to clusterize. The new report
            test_reports (list): list of reports to clusterize. Simulates DB reports (TokenizedReport) elemnets
        Returns:
            pd.DataFrame: list of reports in the same cluster as the new report
        """
        # Obtain tokens of the reports in DB
        reports_db = test_reports if test_reports else get_tokenized_reports_controller()
        # Build dataframe
        df = self.__build_reports_dataframe(reports_db)
        

    def retrain(self) -> None:
        pass



class Classifier():
    
    def __init__(self) -> None:
        pass

    def classify(self, text: str) -> list:
        pass
        
    def retrain(self) -> None:
        pass