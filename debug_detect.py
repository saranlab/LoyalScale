import os
import sys
import pandas as pd
import io

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from src.nlp_mapper import detect_industry, INDUSTRY_SCHEMAS, SYNONYMS, clean_name, get_similarity_score
from sklearn.feature_extraction.text import TfidfVectorizer

def debug_detect_industry(headers: list):
    scores = {}
    
    # Fit TF-IDF on all schema words to build vocab
    all_vocab = []
    for schema in INDUSTRY_SCHEMAS.values():
        all_vocab.extend([clean_name(col) for col in schema])
    for syns in SYNONYMS.values():
        all_vocab.extend([clean_name(syn) for syn in syns])
    
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    vectorizer.fit(all_vocab + [clean_name(h) for h in headers])
    
    for industry, standard_cols in INDUSTRY_SCHEMAS.items():
        matched_count = 0
        total_score = 0.0
        details = []
        
        for u_col in headers:
            best_sim = 0.0
            best_col = None
            for s_col in standard_cols:
                sim = get_similarity_score(u_col, s_col, vectorizer)
                if sim > best_sim:
                    best_sim = sim
                    best_col = s_col
            
            # Count it if it is a reasonable match (sim > 0.40)
            if best_sim > 0.40:
                matched_count += 1
                total_score += best_sim
                details.append((u_col, best_col, round(best_sim, 3)))
                
        # Weigh industry based on proportion of features matched
        prop = matched_count / len(standard_cols)
        score = total_score * prop
        scores[industry] = (score, matched_count, total_score, prop, details)
        
    for ind, (score, m_cnt, t_score, prop, details) in sorted(scores.items(), key=lambda x: x[1][0], reverse=True):
        print(f"Industry: {ind.upper()} | Score: {score:.4f} (Matches: {m_cnt}/{len(INDUSTRY_SCHEMAS[ind])}, Total Sim: {t_score:.2f}, Prop: {prop:.3f})")
        # Print top 5 matches
        print("  Top matches:", details[:5])

if __name__ == '__main__':
    filepath = os.path.join(CURRENT_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(filepath, nrows=2)
    debug_detect_industry(df.columns.tolist())
