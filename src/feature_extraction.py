from sklearn.feature_extraction.text import TfidfVectorizer


def get_feature_extraction(method: str):
    if method == "TfidfVectorizer":
        return TfidfVectorizer()
    else:
        raise Exception("Method is not availble.")