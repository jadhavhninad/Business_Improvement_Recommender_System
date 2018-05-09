class Settings:
    def __init__(self):
        pass

    DATASET_FILE = '../tempe_review_1600.json'
    MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
    REVIEWS_DATABASE = "yelp_clean"
    TAGS_DATABASE = "Tags"
    REVIEWS_COLLECTION = "Reviews"
    CORPUS_COLLECTION = "Corpus"
    RAW_DATABASE = "yelp_raw"
    YELP_RAW_REVIEWS_COLLECTION = "reviews"
    YELP_RAW_BUSINESS_COLLECTION = "business"