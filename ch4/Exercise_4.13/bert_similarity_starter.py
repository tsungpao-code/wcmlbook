from transformers import AutoModel , AutoTokenizer , BertModel ,BertTokenizer
import torch
# Load BERT model and t o k e n i z e r
BERT_PATH = './BERT'
tokenizer = BertTokenizer . from_pretrained ( BERT_PATH )
model = BertModel . from_pretrained ( BERT_PATH )

# S ent e n c e s t o compare
sentence_a = "She enjoys reading books."
sentence_b = "She loves to read."

# T o k e n i z at i o n and model p r o c e s s i n g
def get_sentence_vector(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch . no_grad ():
        outputs = model (** inputs )
        # E x t r a c t [ CLS ] t o k en embedding
        return outputs . last_hidden_state [0, 0]

# Get s e nt e n c e em b eddin g s
vector_a = get_sentence_vector(sentence_a)
vector_b = get_sentence_vector(sentence_b)

# No rmali z e v e c t o r s
norm_a = torch . linalg . norm ( vector_a )
norm_b = torch . linalg . norm ( vector_b )
norm_vector_a = vector_a / norm_a
norm_vector_b = vector_b / norm_b

# C a l c u l a t e c o s i n e s i m i l a r i t y
similarity = torch .dot( norm_vector_a , norm_vector_b ). item ()

print(f"Sentence A: '{sentence_a}'")
print(f"Sentence B: '{sentence_b}'")
print(f"\nSemantic similarity (cosine): {similarity:.4f}")