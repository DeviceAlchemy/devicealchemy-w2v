"""
Scientific Word2Vec Model Inference v1.0
Developed by: DeviceAlchemy LLC
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
"""


from gensim.models import Word2Vec
  
# Load your model
model = Word2Vec.load("models/devicealchemy_w2v.model")


# Populate top 10 similar words
similar_results = model.wv.most_similar("topological_insulator", topn=10)

print(f"\n\n--- Top 10 Words Similar to topological_insulator ---")
for i, (word, score) in enumerate(similar_results[:100]): # Show top 50
    print(f"{i+1}. {word:<15} ({score})")
    

# Ask analogy questions
# Q: Semiconductor is to Si, transition metal is to what?
# A: Ru
analogy_result = model.wv.most_similar(positive=["si", "transition_metal"], negative=["semiconductor"], topn=5)

print("\n\n")
for i, (word, score) in enumerate(analogy_result[:100]): # Show top 50
    print(f"Semiconductor is to Si, transition metal to {word}")



#Sanity check for different materials category
print("\n\n")
wv = model.wv
test_terms = [
    # Battery / solid-state
    "LiFePO4", "cathode", "electrolyte", "solid_state_electrolyte",
    # Semiconductor / thin-film
    "perovskite", "bandgap", "hafnium_oxide",
    # Spintronics (relevant to your domain)
    "CoFeB", "spin_torque", "TMR",
    # Thermoelectrics
    "thermoelectric", "Seebeck",
]
found = 0
for term in test_terms:
    t = term.lower()
    if t in wv:
        similar = wv.most_similar(t, topn=5)
        pairs   = ", ".join(f"{w} ({s:.3f})" for w, s in similar)
        print(f"  {term:35s} -> {pairs}")
        found += 1
    else:
        print(f"  {term:35s} -> NOT IN VOCABULARY")

coverage = found / len(test_terms) * 100
print(f"\n  Vocabulary coverage of test terms: {found}/{len(test_terms)} ({coverage:.0f}%)")
if coverage < 60:
    print("Low coverage - check min_count setting or corpus preprocessing.")
            