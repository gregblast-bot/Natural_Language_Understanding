# This is for INFSCI 2440 in Spring 2026
# Please add comments with your code
# Task 2: Find similar opinions 

import gensim.models.keyedvectors as word2vec

class FindSimilarOpinions:
    extracted_opinions = {}
    word2VecObject = []
    cosine_sim = 0

    def __init__(self, input_cosine_sim, input_extracted_ops):
        self.cosine_sim = input_cosine_sim
        self.extracted_opinions = input_extracted_ops
        word2vec_add = "assign4_word2vec_for_python.bin"
        self.word2VecObject = word2vec.KeyedVectors.load_word2vec_format(word2vec_add, binary=True)
        return

    def get_word_sim(self, word_1, word_2):
        return self.word2VecObject.similarity(word_1, word_2)

    def findSimilarOpinions(self, query_opinion):
        # example data, which you will need to remove in your real code. Only for demo.
        # example_similarity = self.get_word_sim("great", "good")
        # print("Similarity of 'great' and 'good' is " + str(example_similarity))
        # similar_opinions = {'service, good': [1, 2, 3], 'service, excellent': [11, 12]}
        # return similar_opinions
        # 1. Split the query into attribute and assessment
        q_attr, q_assess = [x.strip() for x in query_opinion.split(",")]
        similar_opinions = {}


        # 2. Iterate through all opinions extracted in Task 1
        for ext_opinion, review_ids in self.extracted_opinions.items():
            e_attr, e_assess = [x.strip() for x in ext_opinion.split(",")]

            try:
                # Check similarity of the noun part AND the adjective part
                # 'service' vs 'waiter' should be high; 'good' vs 'excellent' should be high
                if (self.get_word_sim(q_attr, e_attr) >= self.cosine_sim and 
                    self.get_word_sim(q_assess, e_assess) >= self.cosine_sim):
                    similar_opinions[ext_opinion] = review_ids
            except KeyError:
                # Handle cases where words like 'devonshire' aren't in the word2vec model
                if q_attr == e_attr and q_assess == e_assess:
                    similar_opinions[ext_opinion] = review_ids
            # try:
            #     # 3. Calculate similarities for both parts of the opinion
            #     attr_sim = self.get_word_sim(q_attr, e_attr)
            #     assess_sim = self.get_word_sim(q_assess, e_assess)

            #     # 4. Filter based on the tuned cosine_sim threshold
            #     # Check if both the attribute and the assessment are semantically similar
            #     if attr_sim >= self.cosine_sim and assess_sim >= self.cosine_sim:
            #         similar_opinions[ext_opinion] = review_ids
            
            # except KeyError:
            #     # Handle cases where a word is not in the Word2Vec dictionary
            #     # If they are an exact string match, we can still count them as similar
            #     if q_attr == e_attr and q_assess == e_assess:
            #         similar_opinions[ext_opinion] = review_ids
            #     continue

        return similar_opinions