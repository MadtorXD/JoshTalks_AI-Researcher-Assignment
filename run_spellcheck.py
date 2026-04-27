import pandas as pd
import re
from spylls.hunspell import Dictionary
import time

print("Loading dataset...")
df = pd.read_csv("q3_word_list.csv")

# Ensure all values are treated as strings
words = df['word'].astype(str).tolist()

print("Loading dictionary...")
dictionary = Dictionary.from_files('hi_IN')

def clean_word(w):
    w = w.strip()
    # Strip any non-alphanumeric or punctuation that isn't central to the word
    w = re.sub(r'^[\W_]+|[\W_]+$', '', w)
    return w

results = []
correct_count = 0

print(f"Checking {len(words)} words...")
start = time.time()

for w in words:
    cw = clean_word(w)
    status = 'incorrect spelling'
    
    if not cw:
        status = 'correct spelling' # Punctuation only
    elif cw.isnumeric():
        status = 'correct spelling'
    elif len(cw) > 50:
        status = 'incorrect spelling' # Obvious garbage/run-on string
    else:
        # Check Hunspell
        try:
            if dictionary.lookup(cw):
                status = 'correct spelling'
        except Exception:
            pass # fallback to incorrect if un-parseable string crash
    
    if status == 'correct spelling':
        correct_count += 1
        
    results.append(status)

end = time.time()
print(f"Processed {len(words)} in {end-start:.2f} seconds.")

df['spelling_status'] = results

print(f"\n=========================================")
print(f"Final number of unique correct spelled words: {correct_count}")
print(f"=========================================\n")

output_file = "q3_spelling_results.csv"
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Saved results to {output_file}")
