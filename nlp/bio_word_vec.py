from gensim.models import KeyedVectors
filename = '/home/mimic/Downloads/BioWordVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin' 
model = KeyedVectors.load_word2vec_format(filename, binary=True)
vector = model.get_vector('influenza')
print(vector)
