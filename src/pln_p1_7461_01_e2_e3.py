#!/usr/bin/env python3
"""pln_p1_7461_01_e2_e3.py

Ejercicios 2 y 3: Amazon Corpus Reader, preprocesamiento
y análisis básico del corpus.
"""

from pathlib import Path
from collections import Counter
import html
import json
import re

import nltk
from nltk.corpus import stopwords
from nltk.corpus.reader import CorpusReader, CategorizedCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize


# CONFIGURACIÓN

CORPUS_ROOT = Path("corpus_amazon").resolve()

# Añadimos la carpeta que contiene "corpus_amazon"
# como raíz válida para los recursos locales de NLTK.
nltk.data.path.append(str(CORPUS_ROOT.parent))


# FLAGS DE PREPROCESAMIENTO

REMOVE_HTML = True
REMOVE_EMPTY = True
REMOVE_EMOJIS = True
REMOVE_SPECIAL_CHARS = True
NORMALIZE_REPEATED_CHARS = True
REMOVE_STOPWORDS = True
LOWERCASE = True

# AMAZON CORPUS READER

class AmazonCorpusReader(CategorizedCorpusReader, CorpusReader):
    """Lector del corpus de reseñas de Amazon."""

    def __init__(self, root):

        fileids = r"(negative|neutral|positive)/.*\.json"

        kwargs = {
            "cat_pattern": r"(negative|neutral|positive)/.*\.json"
        }

        CorpusReader.__init__(
            self,
            str(root),
            fileids,
            encoding="utf-8"
        )

        CategorizedCorpusReader.__init__(
            self,
            kwargs
        )

        # Stopwords inglesas.
        # Solo se utilizarán cuando REMOVE_STOPWORDS sea True.
        if REMOVE_STOPWORDS:
            self.stop_words = set(stopwords.words("english"))
        else:
            self.stop_words = set()

    def _resolve_fileids(self, fileids=None, categories=None):
        """Determina qué documentos del corpus se deben utilizar."""

        if fileids is not None and categories is not None:
            raise ValueError("No se pueden especificar fileids y categories a la vez.")

        # Selección por categoría
        if categories is not None:
            return self.fileids(categories=categories)

        # Si no se especifica nada, se utiliza todo el corpus
        if fileids is None:
            return self.fileids()

        # Si se proporciona un único fileid
        if isinstance(fileids, str):
            return [fileids]

        # Si se proporciona una colección de fileids
        return list(fileids)

    def _load_json(self, fileid):
        """Carga un documento JSON del corpus."""
        with self.open(fileid) as f:
            return json.load(f)


    def docs(self, fileids=None, categories=None):
        """Devuelve los documentos completos seleccionados."""
        selected_fileids = self._resolve_fileids(
            fileids=fileids,
            categories=categories
        )
        return [
            self._load_json(fileid)
            for fileid in selected_fileids
        ]

    def raw(self, fileids=None, categories=None):
        """Devuelve el texto original de las reseñas seleccionadas."""
        selected_fileids = self._resolve_fileids(
            fileids=fileids,
            categories=categories
        )

        return "\n".join(
            self._load_json(fileid).get("text", "")
            for fileid in selected_fileids
        )


    def rating(self, fileid):
        """Devuelve el rating de una única reseña."""
        return self._load_json(fileid).get("rating")

    def ratings(self, fileids=None, categories=None):
        """Devuelve los ratings de las reseñas seleccionadas."""
        selected_fileids = self._resolve_fileids(
            fileids=fileids,
            categories=categories
        )
        return [
            self._load_json(fileid).get("rating")
            for fileid in selected_fileids
        ]

    def _remove_html(self, text):
        """Elimina etiquetas HTML del texto."""
        # Convierte entidades como &amp; en &
        text = html.unescape(text)
        # Elimina etiquetas como <br>, <p>, etc.
        return re.sub(r"<[^>]+>", " ", text)

    def _remove_emojis(self, text):
        """Elimina emojis y símbolos Unicode."""
        emoji_pattern = re.compile(
            "["
            "\U0001F300-\U0001F5FF"
            "\U0001F600-\U0001F64F"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA70-\U0001FAFF"
            "\u2600-\u26FF"
            "\u2700-\u27BF"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(" ", text)

    def _remove_special_chars(self, text):
        """Elimina símbolos especiales (sin contar letras, números, espacios y signos de puntuación)."""
        return re.sub(
            r"[^\w\s.,!?;:'\"()-]",
            " ",
            text,
            flags=re.UNICODE
        )

    def _normalize_repeated_chars(self, text):
        """Reduce caracteres repetidos tres o más veces."""
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    def _normalize_spaces(self, text):
        """Normaliza espacios y tabulaciones repetidos."""
        return re.sub(r"[ \t]+", " ", text).strip()

    def _preprocess_text(self, text):
        """Aplica al texto las fases de preprocesamiento activadas por las flags."""
        if not isinstance(text, str):
            return ""
        
        if REMOVE_HTML:
            text = self._remove_html(text)

        if REMOVE_EMOJIS:
            text = self._remove_emojis(text)

        if REMOVE_SPECIAL_CHARS:
            text = self._remove_special_chars(text)

        if NORMALIZE_REPEATED_CHARS:
            text = self._normalize_repeated_chars(text)

        if LOWERCASE:
            text = text.lower()

        # La normalización de espacios se realiza después
        # de las operaciones anteriores.
        text = self._normalize_spaces(text)
        return text

    def _preprocess_words(self, words):
        """Aplica las fases de preprocesamiento que deben realizarse tras tokenizar."""
        processed_words = []
        for word in words:
            # Eliminamos tokens vacíos
            if REMOVE_EMPTY and not word.strip():
                continue

            # Eliminamos stopwords si el flag está activado
            # menos las negaciones "not", "no" y "nor" por si son importantes para el análisis de sentimiento.
            if REMOVE_STOPWORDS:
                negations = {"not", "no", "nor"}

                if word.lower() in self.stop_words and word.lower() not in negations:
                    continue

            processed_words.append(word)
        return processed_words

    def paras(self, fileids=None, categories=None):
        """
        Devuelve los párrafos de las reseñas seleccionadas.

        Cada párrafo contiene una lista de oraciones y cada
        oración se representa mediante una lista de tokens.
        """
        selected_fileids = self._resolve_fileids(
            fileids=fileids,
            categories=categories
        )
        paragraphs = []
        for fileid in selected_fileids:
            # Obtenemos primero el texto original.
            raw_text = self._load_json(fileid).get("text", "")

            # Separamos los párrafos antes de normalizar espacios,
            # porque los saltos de línea contienen información
            # sobre la estructura del documento.
            document_paragraphs = re.split(
                r"\n\s*\n",
                raw_text
            )

            for paragraph in document_paragraphs:

                paragraph = paragraph.strip()

                if REMOVE_EMPTY and not paragraph:
                    continue

                paragraph = self._preprocess_text(paragraph)

                if REMOVE_EMPTY and not paragraph:
                    continue

                tokenized_sentences = []

                for sentence in sent_tokenize(paragraph):

                    if REMOVE_EMPTY and not sentence.strip():
                        continue

                    tokens = word_tokenize(sentence)
                    tokens = self._preprocess_words(tokens)

                    if REMOVE_EMPTY and not tokens:
                        continue

                    tokenized_sentences.append(tokens)

                if tokenized_sentences:
                    paragraphs.append(tokenized_sentences)

        return paragraphs

    def sents(self, fileids=None, categories=None):
        """Devuelve las oraciones (listas de tokens) de las reseñas seleccionadas."""
        selected_fileids = self._resolve_fileids(
            fileids=fileids,
            categories=categories
        )
        sentences = []
        for fileid in selected_fileids:

            text = self._load_json(fileid).get("text", "")

            # Aplicamos el preprocesamiento correspondiente
            # antes de segmentar las oraciones.
            text = self._preprocess_text(text)

            for sentence in sent_tokenize(text):
                if REMOVE_EMPTY and not sentence.strip():
                    continue

                tokens = word_tokenize(sentence)
                tokens = self._preprocess_words(tokens)

                if REMOVE_EMPTY and not tokens:
                    continue

                sentences.append(tokens)

        return sentences

    def words(self, fileids=None, categories=None):
        """Devuelve las palabras/tokens de las reseñas seleccionadas."""
        selected_fileids = self._resolve_fileids(
            fileids=fileids,
            categories=categories
        )
        words = []
        for fileid in selected_fileids:

            text = self._load_json(fileid).get("text", "")
            text = self._preprocess_text(text)

            tokens = word_tokenize(text)
            tokens = self._preprocess_words(tokens)

            words.extend(tokens)
        return words

    def describe(self, fileids=None, categories=None):
            """Calcula estadísticas generales de los documentos seleccionados."""

            # Seleccionamos los documentos que queremos analizar
            fileids = self._resolve_fileids(
                fileids=fileids,
                categories=categories
            )

            # Documentos
            total_documents = len(fileids)

            documents_by_category = {
                category: 0
                for category in self.categories()
            }

            for fileid in fileids:
                category = fileid.split("/")[0]
                if category in documents_by_category:
                    documents_by_category[category] += 1

            # Información de los documentos
            reviews_with_rating = 0
            reviews_with_text = 0
            reviews_with_rating_and_text = 0

            rating_values = []

            users = set()
            products = set()

            for fileid in fileids:

                document = self._load_json(fileid)

                rating = document.get("rating")
                text = document.get("text", "")
                user_id = document.get("user_id")
                product_id = document.get("parent_asin")

                has_rating = rating is not None
                has_text = isinstance(text, str) and bool(text.strip())

                if has_rating:
                    reviews_with_rating += 1
                    rating_values.append(rating)

                if has_text:
                    reviews_with_text += 1

                if has_rating and has_text:
                    reviews_with_rating_and_text += 1

                if user_id:
                    users.add(user_id)

                if product_id:
                    products.add(product_id)

            # Distribución de ratings
            rating_distribution = dict(
                sorted(Counter(rating_values).items())
            )

            # Párrafos, oraciones y palabras de los documentos seleccionados
            paragraphs = self.paras(fileids=fileids)
            sentences = self.sents(fileids=fileids)
            corpus_words = self.words(fileids=fileids)

            # Vocabulario:
            # utilizamos minúsculas para no distinguir entre palabras
            # iguales con diferente capitalización.
            vocabulary = {
                word.lower()
                for word in corpus_words
                if any(character.isalnum() for character in word)
            }

            # Resultado
            return {
                "documents": total_documents,
                "documents_by_category": documents_by_category,
                "reviews_with_rating": reviews_with_rating,
                "reviews_with_text": reviews_with_text,
                "reviews_with_rating_and_text": reviews_with_rating_and_text,
                "rating_distribution": rating_distribution,
                "paragraphs": len(paragraphs),
                "sentences": len(sentences),
                "words": len(corpus_words),
                "vocabulary_size": len(vocabulary),
                "unique_users": len(users),
                "unique_products": len(products)
            }


# ============================================================
# PRUEBAS DEL CORPUS
# ============================================================

if __name__ == "__main__":

    corpus = AmazonCorpusReader(CORPUS_ROOT)

    # ========================================================
    # EJERCICIO 2
    # ========================================================

    print("Categorías:")
    print(corpus.categories())

    print("\nNúmero total de documentos (reviews):")
    print(len(corpus.fileids()))

    print("\nDocumentos por categoría:")

    for category in corpus.categories():
        files = corpus.fileids(categories=category)
        print(f"{category}: {len(files)}")

    # --------------------------------------------------------
    # Documento individual
    # --------------------------------------------------------

    fileid = corpus.fileids(categories="positive")[0]

    print("\n--- PRUEBA DE UN DOCUMENTO ---")

    print("\nFileid:")
    print(fileid)

    print("\nCategoría:")
    print(corpus.categories(fileid))

    print("\nRating:")
    print(corpus.rating(fileid))

    print("\nTexto en crudo:")
    print(corpus.raw(fileids=fileid))

    print("\nPárrafos:")
    print(corpus.paras(fileids=fileid))

    print("\nOraciones:")
    print(corpus.sents(fileids=fileid))

    print("\nPalabras:")
    print(corpus.words(fileids=fileid))

    # --------------------------------------------------------
    # Categoría
    # --------------------------------------------------------

    print("\n--- PRUEBA POR CATEGORÍA ---")

    print("\nNúmero de palabras positivas:")
    print(len(corpus.words(categories="positive")))

    print("\nNúmero de oraciones neutrales:")
    print(len(corpus.sents(categories="neutral")))

    print("\nPrimeros 10 ratings negativos:")
    print(corpus.ratings(categories="negative")[:10])

    # --------------------------------------------------------
    # Corpus completo
    # --------------------------------------------------------

    print("\n--- PRUEBA DEL CORPUS COMPLETO ---")

    print("\nNúmero total de palabras:")
    print(len(corpus.words()))

    print("\nNúmero total de oraciones:")
    print(len(corpus.sents()))

    print("\nNúmero total de párrafos:")
    print(len(corpus.paras()))

    print("\nNúmero total de ratings:")
    print(len(corpus.ratings()))

    # ========================================================
    # EJERCICIO 3
    # ========================================================

    print("\n--- ESTADÍSTICAS DEL CORPUS ---")

    stats = corpus.describe()

    print("\nNúmero total de documentos:")
    print(stats["documents"])

    print("\nDocumentos por categoría:")
    for category, count in stats["documents_by_category"].items():
        print(f"{category}: {count}")

    print("\nReseñas con rating:")
    print(stats["reviews_with_rating"])

    print("\nReseñas con texto:")
    print(stats["reviews_with_text"])

    print("\nReseñas con rating y texto:")
    print(stats["reviews_with_rating_and_text"])

    print("\nDistribución de ratings:")
    for rating, count in stats["rating_distribution"].items():
        print(f"{rating}: {count}")

    print("\nNúmero de párrafos:")
    print(stats["paragraphs"])

    print("\nNúmero de oraciones:")
    print(stats["sentences"])

    print("\nNúmero total de palabras/tokens:")
    print(stats["words"])

    print("\nNúmero de palabras distintas (vocabulario):")
    print(stats["vocabulary_size"])

    print("\nNúmero de usuarios distintos:")
    print(stats["unique_users"])

    print("\nNúmero de productos distintos:")
    print(stats["unique_products"])

    # Prueba de describe con categoría específica
    print("\n--- DESCRIBE: RESEÑAS POSITIVAS ---")
    stats_positive = corpus.describe(categories="positive")

    print("\nNúmero total de documentos positivos:")
    print(stats_positive["documents"])

    print("\nReseñas con rating:")
    print(stats_positive["reviews_with_rating"])

    print("\nReseñas con texto:")
    print(stats_positive["reviews_with_text"])

    print("\nReseñas con rating y texto:")
    print(stats_positive["reviews_with_rating_and_text"])

    print("\nNúmero de párrafos:")
    print(stats_positive["paragraphs"])

    print("\nNúmero de oraciones:")
    print(stats_positive["sentences"])

    print("\nNúmero total de palabras/tokens:")
    print(stats_positive["words"])

    print("\nNúmero de palabras distintas (vocabulario):")
    print(stats_positive["vocabulary_size"])

    print("\nNúmero de usuarios distintos:")
    print(stats_positive["unique_users"])

    print("\nNúmero de productos distintos:")
    print(stats_positive["unique_products"])

    # Prueba de describe con un subconjunto de documentos
    print("\n--- DESCRIBE: 10 DOCUMENTOS ---")
   
    print("File IDs de los primeros 10 documentos:")
    diez_fileids = corpus.fileids()[:10]
    print(diez_fileids)
    
    stats_10 = corpus.describe(fileids=diez_fileids)
    
    print("\nNúmero total de documentos:")
    print(stats_10["documents"])

    print("\nDocumentos por categoría:")
    for category, count in stats_10["documents_by_category"].items():
        print(f"{category}: {count}")

    print("\nReseñas con rating:")
    print(stats_10["reviews_with_rating"])

    print("\nReseñas con texto:")
    print(stats_10["reviews_with_text"])

    print("\nReseñas con rating y texto:")
    print(stats_10["reviews_with_rating_and_text"])

    print("\nDistribución de ratings:")
    for rating, count in stats_10["rating_distribution"].items():
        print(f"{rating}: {count}")

    print("\nNúmero de párrafos:")
    print(stats_10["paragraphs"])

    print("\nNúmero de oraciones:")
    print(stats_10["sentences"])

    print("\nNúmero total de palabras/tokens:")
    print(stats_10["words"])

    print("\nNúmero de palabras distintas (vocabulario):")
    print(stats_10["vocabulary_size"])

    print("\nNúmero de usuarios distintos:")
    print(stats_10["unique_users"])

    print("\nNúmero de productos distintos:")
    print(stats_10["unique_products"])