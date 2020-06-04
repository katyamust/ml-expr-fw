from seqeval.metrics import f1_score, accuracy_score

from ner_sample.evaluation import Evaluator, NEREvaluationMetrics


class NEREvaluator(Evaluator):
    """
    This class holds the logic for evaluating a NER prediction outcome
    """

    def evaluate(self, predicted_sentences) -> NEREvaluationMetrics:
        golds = []
        predicted = []
        print(predicted_sentences)
        for sentence in predicted_sentences:
            gold_tags = [token.get_tag("gold_ner").value for token in sentence.tokens]
            golds.append(gold_tags)
            predicted_tags = [token.get_tag("ner").value for token in sentence.tokens]
            predicted.append(predicted_tags)

        f1 = f1_score(golds, predicted)
        accuracy = accuracy_score(golds, predicted)
        return NEREvaluationMetrics(f1=f1, accuracy=accuracy)
