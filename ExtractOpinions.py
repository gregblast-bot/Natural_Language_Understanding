# This is for INFSCI 2440 in Spring 2026
# Please add comments with your code
# Task 1: Extract opinion 

from StanfordNLP import StanfordNLP

'''
This class is responsible for extracting opinions from reviews using the StanfordNLP client.
'''
class ExtractOpinions:
    # Extracted opinions and corresponding review id is saved in extracted_pairs, where KEY is the opinion and VALUE
    # is the set of review_ids where the opinion is extracted from.
    # Opinion should in form of "attribute, assessment", such as "service, good".
    extracted_opinions = {}

    '''
    The constructor initializes the extracted_opinions dictionary and creates an instance of the StanfordNLP client.
    '''
    def __init__(self):
        # Initialize the storage for extracted opinions and the StanfordNLP client
        self.extracted_opinions = {}
        self.sNLP = StanfordNLP()

    '''
    This method takes a review ID and the review content, 
    processes it using the StanfordNLP client to extract opinions based on specific dependency patterns, 
    and stores them in the extracted_opinions dictionary.

    Args:
        review_id (str): The ID of the review.
        review_content (str): The content of the review to be processed.
    '''
    def extract_pairs(self, review_id, review_content):
        # Annotate the review content to get tokens and dependencies, CoreNLP returns annotations as result['sentences'][0]['tokens'].
        annotation = self.sNLP.annotate(review_content)
        
        # Iterate through each sentence in the annotated review.
        for sentence in annotation['sentences']:
            # Convert tokens list to a dictionary for easy O(1) lookup by index.
            tokens = StanfordNLP.tokens_to_dict(sentence['tokens'])
            
            # Iterate through the enhanced dependencies, CoreNLP returns dependencies as [relation, governor_index, dependent_index].
            # EnhancedPlusPlusDependenciesAnnotation provides the most semantically rich set of dependencies.
            for dependency in sentence['enhancedPlusPlusDependencies']:
                relationship = dependency['dep']
                governor_idx = dependency['governor']
                dependent_idx = dependency['dependent']
                
                # Relationship: Adjectival Modifier (e.g. "great service").
                if relationship == 'amod':
                    # Attribute is the governor (noun), Value is the dependent (adjective).
                    self.save_quality_opinion(tokens[governor_idx], tokens[dependent_idx], review_id)    
                # Relationship: Nominal Subject (e.g. "service is great").
                elif relationship == 'nsubj':
                    # Attribute is the dependent (noun), Value is the governor (adjective).
                    self.save_quality_opinion(tokens[dependent_idx], tokens[governor_idx], review_id)


    '''
    This helper method checks if the given tokens match the expected POS tags for attributes and assessments,
    lemmatizes them, and saves the opinion in the extracted_opinions dictionary along with the review ID.

    Args:
        attr_token (dict): The token representing the attribute (expected to be a noun).
        assess_token (dict): The token representing the assessment (expected to be an adjective).
        review_id (str): The ID of the review from which the opinion was extracted.
    '''
    def save_quality_opinion(self, attribute_token, value_token, review_id):
        # Check POS tags to ensure we have a Noun (NN) and an Adjective (JJ).
        if attribute_token['pos'].startswith('NN') and value_token['pos'].startswith('JJ'):
            # Use lemmatization to group similar words (e.g. "pricing", "prices", "price").
            attribute = attribute_token['lemma'].lower()
            value = value_token['lemma'].lower()
            
            # Use a list to store multiple review IDs for the same opinion.
            opinion_key = f"{attribute}, {value}"
            if opinion_key not in self.extracted_opinions:
                self.extracted_opinions[opinion_key] = []
            
            # Only add the review ID if it's not already there, this prevents duplicates.
            if review_id not in self.extracted_opinions[opinion_key]:
                self.extracted_opinions[opinion_key].append(review_id)

