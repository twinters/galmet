from galmet import genetictransformer


class BaselineTextModifier:
    """ Class that just substitutes a certain number of tokens, as baseline to compare with GALMET """

    def __init__(self, masker: genetictransformer.MaskPredictor, allowed_edits=5):
        self.masker = masker
        self.allowed_edits = allowed_edits

    # Create mutation function
    def modify_sentence(self, sentence):
        tokenized = self.masker.tokenize_sentence(sentence)
        for i in range(self.allowed_edits):
            tokenized = self.masker.mutate_token_randomly(tokenized, allow_same=False)
        return self.masker.detokenize_sentence(tokenized)
