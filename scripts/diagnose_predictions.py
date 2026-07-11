import pandas as pd
import re
from collections import Counter

def main():
    csv_path = 'output/vision_t5_raw.csv'
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error opening {csv_path}: {e}")
        return

    print("=== NeSy-CARE Prediction Diagnostics ===")
    print(f"Total test examples: {len(df)}")
    
    # 1. Length analysis
    pred_lens = df['prediction'].fillna('').apply(lambda x: len(x.split()))
    ref_lens = df['reference'].fillna('').apply(lambda x: len(x.split()))
    print(f"Average prediction length: {pred_lens.mean():.1f} words")
    print(f"Average reference length:  {ref_lens.mean():.1f} words")
    
    # 2. Vocabulary analysis
    pred_words = [w for p in df['prediction'].fillna('') for w in re.findall(r'\w+', p.lower())]
    ref_words = [w for r in df['reference'].fillna('') for w in re.findall(r'\w+', r.lower())]
    
    print(f"Unique words in predictions: {len(set(pred_words))}")
    print(f"Unique words in references:  {len(set(ref_words))}")
    
    # 3. Check for repetitive loops (common T5 failure mode)
    loops = 0
    for p in df['prediction'].fillna(''):
        words = re.findall(r'\w+', p.lower())
        if len(words) > 10:
            # Check 3-gram counts
            tg = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            c = Counter(tg)
            if c and c.most_common(1)[0][1] > 3:
                loops += 1
    print(f"Reports with repetitive loops (>3 identical 3-grams): {loops} ({loops/len(df)*100:.1f}%)")
    
    # 4. Check edit prompt leakage (did T5 output prompt metadata?)
    prompt_leak = 0
    for p in df['prediction'].fillna(''):
        if 'template:' in p.lower() or 'edit:' in p.lower() or 'generate report:' in p.lower():
            prompt_leak += 1
    print(f"Reports leaking prompt metadata ('template:', 'edit:'): {prompt_leak} ({prompt_leak/len(df)*100:.1f}%)")
    
    # 5. Print a few sample pairs for manual inspection
    print("\n=== Random Sample Inspection ===")
    sample_df = df.sample(min(5, len(df)), random_state=42)
    for idx, row in sample_df.iterrows():
        print(f"Study ID: {row['study_id']}")
        print(f"  [Ref]:  {row['reference'].strip()}")
        print(f"  [Pred]: {row['prediction'].strip()}")
        print("-" * 40)

if __name__ == '__main__':
    main()
