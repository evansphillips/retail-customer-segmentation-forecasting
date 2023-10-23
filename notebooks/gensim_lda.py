from typing import List, Type, Tuple, Union
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phraser, Phrases
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
import spacy
import pandas as pd


def sent_to_words(sentences: List[str]) -> List[List[str]]:
    """
    Tokenize each sentence into a list of words using simple_preprocess from gensim.

    Parameters:
        sentences (List[str]): List of sentences to be tokenized.

    Returns:
        List[List[str]]: List of lists, where each inner list contains the tokenized words of a sentence.
    """
    for sentence in sentences:
        yield simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(texts: List[List[str]], stop_words: list) -> List[List[str]]:
    """
    Remove stopwords from a list of tokenized texts.

    Parameters:
        texts (List[List[str]]): List of tokenized texts.
        stop_words (list): a list of stop words to remove.

    Returns:
        List[List[str]]: List of tokenized texts with stopwords removed.
    """
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


def make_bigrams(texts: List[List[str]], bigram_mod: Phraser) -> List[List[str]]:
    """
    Create bigrams from a list of tokenized texts.

    Parameters:
        texts (List[List[str]]): List of tokenized texts.
        bigram_mod (Phraser): Phraser object that takes in Phrases object of a bigram.

    Returns:
        List[List[str]]: List of texts with bigrams.
    """
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(
    texts: List[List[str]], bigram_mod: Phraser, trigram_mod: Phraser
) -> List[List[str]]:
    """
    Create trigrams from a list of tokenized texts.

    Parameters:
        texts (List[List[str]]): List of tokenized texts.
        bigram_mod (Phraser): Phraser object that takes in Phrases object of a bigram.
        trigram_mod (Phraser): Phraser object that takes in Phrases object of a trigram.

    Returns:
        List[List[str]]: List of texts with trigrams.
    """
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(
    texts: List[List[str]],
    nlp: Type["spacy.lang.en.English"],
    allowed_postags: List[str] = ["NOUN", "ADJ", "VERB", "ADV"],
) -> List[List[str]]:
    """
    Lemmatize a list of tokenized texts.

    Parameters:
        texts (List[List[str]]): List of tokenized texts.
        nlp (spacy.lang.en.English): spacy 'en_core_web_sm' model, keeping only tagger component.
        allowed_postags (List[str]): List of allowed POS tags for lemmatization.

    Returns:
        List[List[str]]: List of lemmatized texts.
    """
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


def compute_coherence_values(
    dictionary: dict,
    corpus: list,
    texts: list,
    limit: int,
    id2word: corpora.Dictionary,
    start: int = 2,
    step: int = 3,
):
    """
    Compute c_v coherence for various number of topics.

    Parameters:
        dictionary (dict): Gensim dictionary.
        corpus (list): Gensim corpus.
        texts (list): List of input texts.
        limit (int): Max number of topics.
        id2word (corpora.Dictionary): corpora.Dictionary object of lemmatized documents (input to LdaModel).
        start (int): Start number of topics.
        step (int): Increment to chagne the amount of topics by.

    Returns:
        model_list (list): List of LDA topic models.
        coherence_values (list): Coherence values corresponding to the LDA model with respective number of topics.
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def format_topics_sentences(
    texts: List[str],
    ldamodel: LdaModel = None,
    corpus: List[List[Tuple[int, int]]] = None  
) -> pd.DataFrame:
    """
    Outputs a pd.DataFrame that displays the main topic in each document.

    Parameters:
        texts (list): List of unique descriptions.
        ldamodel (gensim.models.ldamodel.LdaModel): Trained gensim LDA model.
        corpus (List[List[Tuple[int, int]]]): Gensim corpus mapping of word_id to word_frequency.

    Returns:
        sent_topics_df (pd.DataFrame): A pd.DataFrame of documents numbers, topics numbers, percent
        contributions of a document to each topic, keywords for the topic, and the original text.
    """
    if ldamodel is None or corpus is None or texts is None:
        raise ValueError("Please provide valid values for ldamodel, corpus, and texts.")

    # Init output
    sent_topics_data = {
        "Dominant_Topic": [],
        "Perc_Contribution": [],
        "Topic_Keywords": [],
        "Text": [],
    }

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution, and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_data["Dominant_Topic"].append(int(topic_num))
                sent_topics_data["Perc_Contribution"].append(round(prop_topic, 4))
                sent_topics_data["Topic_Keywords"].append(topic_keywords)
                sent_topics_data["Text"].append(texts[i])
            else:
                break

    # Create DataFrame
    sent_topics_df = pd.DataFrame(sent_topics_data)

    return sent_topics_df
