from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
class SimpleRAG:
    def __init__(self, docs):
        self.docs=docs; self.vec=TfidfVectorizer(stop_words='english'); self.X=self.vec.fit_transform(docs)
    def retrieve(self,q,k=3):
        qv=self.vec.transform([q]); sims=cosine_similarity(qv,self.X).ravel()
        idx=sims.argsort()[::-1][:k]; return [(int(i), float(sims[i]), self.docs[int(i)]) for i in idx]
