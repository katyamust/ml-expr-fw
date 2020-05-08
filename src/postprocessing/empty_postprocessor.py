from src.postprocessing.data_postprocessor import DataPostprocessor


class EmptyPostprocessor(DataPostprocessor):
    """
    Postprocessor calls that doesn't do anything data .
    """
    def __init__(self):
        super().__init__()

    def postprocess(self, text):
        pass

    def postprocess_as_list(self, texts):
        pass

    def __str__(self):
        return " "

    def __repr__(self):
        return self.__str__()
