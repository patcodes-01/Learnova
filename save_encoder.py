from sentence_transformers import SentenceTransformer

# 1. Download / load pretrained SBERT miniLM
base_model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(base_model_name)

# 2. Save it under your own neutral name
output_dir = "models/pyq_semantic_encoder"
model.save(output_dir)

print("Saved encoder to:", output_dir)
