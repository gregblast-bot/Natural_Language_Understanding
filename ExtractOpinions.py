# This is for INFSCI 2440 in Spring 2026
# Please add comments with your code
# Task 1: Extract opinion 

from StanfordNLP import StanfordNLP

class ExtractOpinions:
    # Extracted opinions and corresponding review id is saved in extracted_pairs, where KEY is the opinion and VALUE
    # is the set of review_ids where the opinion is extracted from.
    # Opinion should in form of "attribute, assessment", such as "service, good".
    extracted_opinions = {}

    def __init__(self):
        # Initialize the storage for extracted opinions
        self.extracted_opinions = {}
        # Initialize the StanfordNLP client
        self.sNLP = StanfordNLP()

    def extract_pairs(self, review_id, review_content):
        # example data, which you will need to remove in your real code. Only for demo.
        #self.extracted_opinions = {'service, good': [1, 2, 5], 'service, excellent': [4, 6]}
        # Step 1: Annotate the review content to get tokens and dependencies
        annotation = self.sNLP.annotate(review_content)
        
        for sentence in annotation['sentences']:
            # Convert tokens list to a dictionary for easy O(1) lookup by index
            tokens = StanfordNLP.tokens_to_dict(sentence['tokens'])
            
            # Step 2: Iterate through the enhanced dependencies
            # CoreNLP returns dependencies as [relation, governor_index, dependent_index]
            for dep in sentence['enhancedPlusPlusDependencies']:
                rel = dep['dep']
                gov_idx = dep['governor']
                dep_idx = dep['dependent']
                
                # Pattern A: Adjectival Modifier (e.g., "great service")
                if rel == 'amod':
                    # Attribute is the governor (noun), Assessment is the dependent (adj)
                    self.save_opinion(tokens[gov_idx], tokens[dep_idx], review_id)
                
                # Pattern B: Nominal Subject (e.g., "service is great")
                elif rel == 'nsubj':
                    # Assessment is the governor (adj), Attribute is the dependent (noun)
                    self.save_opinion(tokens[dep_idx], tokens[gov_idx], review_id)


    def save_opinion(self, attr_token, assess_token, review_id):
        # Check POS tags to ensure we have a Noun (NN) and an Adjective (JJ)
        if attr_token['pos'].startswith('NN') and assess_token['pos'].startswith('JJ'):
            # Use lemmatization to group similar words (e.g., "prices" -> "price")
            attr = attr_token['lemma'].lower()
            assess = assess_token['lemma'].lower()
            
            opinion_key = f"{attr}, {assess}"
            
            # Use a list to store multiple review IDs for the same opinion
            if opinion_key not in self.extracted_opinions:
                self.extracted_opinions[opinion_key] = []
            
            # Only add the ID if it's not already there (prevents duplicates in one review)
            if review_id not in self.extracted_opinions[opinion_key]:
                self.extracted_opinions[opinion_key].append(review_id)

