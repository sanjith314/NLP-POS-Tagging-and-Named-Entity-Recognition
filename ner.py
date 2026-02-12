import pandas as pd
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import sent_tokenize, word_tokenize
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STANFORD_NER_JAR = os.path.join(
    BASE_DIR,
    "stanford-ner-2020-11-17",
    "stanford-ner.jar"
)

STANFORD_NER_MODEL = os.path.join(
    BASE_DIR,
    "stanford-ner-2020-11-17",
    "classifiers",
    "english.muc.7class.distsim.crf.ser.gz"
)



# Function: get_ner(text)
# fname: A string containing text to be processed
# Returns: A list of tuples containing (token, tag)
def get_ner(text, path_to_jar, path_to_model):
    if path_to_jar is None:
        path_to_jar = STANFORD_NER_JAR
    if path_to_model is None:
        path_to_model = STANFORD_NER_MODEL

    ner_tagger = StanfordNERTagger(
        model_filename=path_to_model,
        path_to_jar=path_to_jar
    )

    ner_tags = []

    sentences = sent_tokenize(text)

    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged_sent = ner_tagger.tag(tokens)

        for token, tag in tagged_sent:
            if tag != 'O':
                ner_tags.append((token, tag))

    return ner_tags

if __name__ == "__main__":
    
    path_to_jar = None
    path_to_model = None

    data = pd.read_csv('dataset.csv')

    for index, row in data.iterrows():
        if index >= 2: 
            break

        title = row['title']
        summary = row['summary']
        entities = get_ner(summary,path_to_jar,path_to_model)

        print(f"Movie Title: {title}")
        print(f"List of Named Entities: {entities}")


### Expected Outputs

# Movie Title: Incubus
# List of Named Entities: [('Orin', 'PERSON'), ('Kiefer', 'PERSON'), ('Peter', 'PERSON'), ('Peter', 'PERSON'), ('Jay', 'PERSON'), ('Holly', 'PERSON'), ('Peter', 'PERSON'), ('Holly', 'PERSON'), ('Jay', 'PERSON'), ('Jay', 'PERSON'), ('Peter', 'PERSON'), ('Jay', 'PERSON'), ('Jay', 'PERSON'), ('Jay', 'PERSON')]
# Movie Title: H
# List of Named Entities: [('Shin', 'PERSON'), ('Hyun', 'PERSON'), ('Shin', 'PERSON'), ('Shin', 'PERSON'), ('Chu', 'PERSON'), ('Kang', 'ORGANIZATION'), ('Shin', 'PERSON')]
# Movie Title: Bring Me the Head of Alfredo Garcia
# List of Named Entities: [('Alfredo', 'PERSON'), ('Garcia', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('$', 'MONEY'), ('1', 'MONEY'), ('million', 'MONEY'), ('Alfredo', 'PERSON'), ('Garcia', 'PERSON'), ('Mexico', 'LOCATION'), ('City', 'LOCATION'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('Quill', 'PERSON'), ('Bennie', 'PERSON'), ('United', 'ORGANIZATION'), ('States', 'ORGANIZATION'), ('Army', 'ORGANIZATION'), ('Garcia', 'PERSON'), ('Garcia', 'LOCATION'), ('Elita', 'LOCATION'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), 
# ('Garcia', 'PERSON'), ('Quill', 'PERSON'), ('Max', 'PERSON'), ('$', 'MONEY'), ('10,000', 'MONEY'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Bennie', 'PERSON'), ('Bennie', 'PERSON'), ('Elita', 'PERSON'), ('Elita', 'LOCATION'), ('Bennie', 'PERSON'), ('Kristofferson', 'PERSON'), ('Elita', 'ORGANIZATION'), ('Elita', 'ORGANIZATION'), ('Elita', 'LOCATION'), ('Garcia', 'PERSON'), ('Mexico', 'LOCATION'), ('City', 'LOCATION'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Elita', 'PERSON'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Garcia', 'PERSON'), ('Alfredo', 'PERSON'), 
# ('Elita', 'ORGANIZATION'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Quill', 'PERSON'), ('Garcia', 'PERSON'), ('Quill', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('Bennie', 'PERSON'), ('Garcia', 'PERSON'), ('Mexico', 'LOCATION'), ('City', 'LOCATION'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Garcia', 'PERSON'), ('$', 'MONEY'), ('10,000', 'MONEY'), ('Elita', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('Garcia', 'PERSON'), ('Bennie', 'PERSON'), ('Garcia', 'PERSON'), ('Elita', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION'), ('Bennie', 'PERSON'), ('Bennie', 'PERSON'), ('obliges', 'PERSON'), ('El', 'ORGANIZATION'), ('Jefe', 'ORGANIZATION')]