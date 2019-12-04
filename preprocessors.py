from typing import Optional

from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


@preprocessor()
def get_person_text(cand: DataPoint) -> DataPoint:
    """
    Returns the text for the two person mentions in candidate sentence.
    :param cand: A candidate DF.
    :return: Candidate DF with new column, a list of entity names.
    """
    person_names = []
    for index in [1, 2]:
        field_name = "person{j}_word_idx".format(j=index)
        start = cand[field_name][0]
        end = cand[field_name][1] + 1
        person_names.append(" ".join(cand["tokens"][start:end]))
    cand.person_names = person_names
    return cand


@preprocessor()
def get_text_between(cand: DataPoint) -> DataPoint:
    """
    Returns the text between two entity mentions.
    :param cand: A candidate DF.
    :return: Candidate DF with new column, text between entity mentions.
    """
    start = cand.person1_word_idx[1] + 1
    end = cand.person2_word_idx[0]
    cand.text_between = " ".join(cand.tokens[start:end])
    return cand


@preprocessor()
def get_left_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the three length window to the left of entity mentions.
    :param cand: A candidate DF.
    :return: Candidate DF with two new columns, each a list of tokens to the left of entities.
    """
    # TODO: make window a parameter
    window = 3
    end = cand.person1_word_idx[0]
    cand.person1_left_tokens = cand.tokens[0:end][1 - window: -1]
    end = cand.person2_word_idx[0]
    cand.person2_left_tokens = cand.tokens[0:end][1 - window: -1]
    return cand


@preprocessor()
def get_persons_last_name(cand: DataPoint) -> DataPoint:
    """
    Returns entity last names.
    :param cand: A candidate DF.
    :return: Candidate DF with a new column, a list of last names.
    """
    cand = get_person_text(cand)
    person1_name, person2_name = cand.person_names
    person1_last_name = (person1_name.split(" ")[-1] if len(person1_name.split(" ")) > 1 else None)
    person2_last_name = (person2_name.split(" ")[-1] if len(person2_name.split(" ")) > 1 else None)
    cand.person_lastnames = [person1_last_name, person2_last_name]
    return cand


def last_name(s: str) -> Optional[str]:
    name_parts = s.split(" ")
    return name_parts[-1] if len(name_parts) > 1 else None
