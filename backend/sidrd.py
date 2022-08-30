"""Module for SIDRD funcionality."""
from string import punctuation

import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer


RESOURCES_PATH = 'resources'


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
        self.__URL_FORBIDDEN_CHARS = [c for c in self.__EXTRA_CHARACTERS if c not in [':', '/', '?', '=', '&', '#']]
        self.__CUSTOM_WORDS = ['info', 'https', 'http', 'org', 'com', 'net', 'edu']

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
        if token.isdigit():
            return []

        token = token.lower()

        if token.startswith('http'):
            token = ''.join(
                [w if w not in self._URL_FORBIDDEN_CHARS else '' for w in token]
            )
        else:
            token = ''.join(
                [w if w not in self._EXTRA_CHARACTERS else ' ' for w in token]
            )

        token = token.strip()
        return token.split() if len(token.split()) > 1 else [token]

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

    def __init__(self) -> None:
        pass

    def vectorize(self, text: str) -> list:
        pass

    def retrain(self) -> None:
        pass


class Clusterizer():

    def __init__(self) -> None:
        pass

    def clusterize(self, text: str) -> list:
        pass

    def retrain(self) -> None:
        pass



class Classifier():
    
    def __init__(self) -> None:
        pass

    def classify(self, text: str) -> list:
        pass
        
    def retrain(self) -> None:
        pass