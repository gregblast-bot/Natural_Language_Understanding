# This is for INFSCI 2440 in Spring 2026
# Please add comments with your code
# Task 2: Find similar opinions 

import gensim.models.keyedvectors as word2vec

'''
    A class to find similar opinions based on semantic similarity using Word2Vec.
    It takes a cosine similarity threshold and a dictionary of extracted opinions as input.
'''
class FindSimilarOpinions:
    extracted_opinions = {}
    word2VecObject = []
    cosine_sim = 0

    '''
    The constructor initializes the cosine similarity threshold, extracted opinions, and loads the pre-trained Word2Vec model.

    Args:
        input_cosine_sim (float): The cosine similarity threshold for determining if two opinions are similar.
        input_extracted_ops (dict): A dictionary of extracted opinions where the key is the opinion (in the form "attribute, value") and the value is a list of review IDs where that opinion was extracted.
    '''
    def __init__(self, input_cosine_sim, input_extracted_ops):
        self.cosine_sim = input_cosine_sim
        self.extracted_opinions = input_extracted_ops
        word2vec_add = "assign4_word2vec_for_python.bin"
        self.word2VecObject = word2vec.KeyedVectors.load_word2vec_format(word2vec_add, binary=True)
        return

    '''
    This method calculates the cosine similarity between two words using the loaded Word2Vec model.

    Args:
        word_1 (str): The first word for which to calculate similarity.
        word_2 (str): The second word for which to calculate similarity.

    Returns:
        float: The cosine similarity between the two words.
    '''
    def get_word_sim(self, word_1, word_2):
        return self.word2VecObject.similarity(word_1, word_2)

    '''
    This method finds opinions that are similar to a given query opinion based on the cosine similarity of both the attribute and the value parts of the opinion. 

    Args:
        query_opinion (str): The opinion to compare against the extracted opinions, in the form "attribute, value" (e.g., "service, good").

    Returns:
        dict: A dictionary where the key is a similar opinion and the value is a list of review IDs where that opinion was extracted.
    '''

    def findSimilarOpinions(self, query_opinion):
        # Split the query into the attribute and the value. Initialize dictionary for similar opinions.
        q_attribute, q_value = [x.strip() for x in query_opinion.split(",")]
        similar_opinions = {}

        # Iterate through all opinions extracted in Task 1.
        for extracted_opinion, review_ids in self.extracted_opinions.items():
            e_attribute, e_value = [x.strip() for x in extracted_opinion.split(",")]

            # Try to calculate similarities and handle cases where the opinion might not be in the Word2Vec vocabulary. 
            try:
                # Calculate the similarities for both parts of the opinion.
                attribute_sim = self.get_word_sim(q_attribute, e_attribute)
                value_sim = self.get_word_sim(q_value, e_value)

                # Filter based on the tuned cosine_sim threshold. This checks if both the attribute and the value are semantically similar.
                if attribute_sim >= self.cosine_sim and value_sim >= self.cosine_sim:
                    similar_opinions[extracted_opinion] = review_ids
            except KeyError:
                # If either word is missing, we can skip similarity calculation for that opinion.
                if q_attribute == e_attribute and q_value == e_value:
                    similar_opinions[extracted_opinion] = review_ids
                continue

        return similar_opinions