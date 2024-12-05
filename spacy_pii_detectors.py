from typing import List
import spacy
import re

nlp_eng = spacy.load("en_core_web_trf")

def split_long_text(text: str) -> List[str]:
    special_sep = "_/\SEP/\_"
    def _splitter(match_object):
        return match_object.group(0)+special_sep
    
    texts = [text]
    #if len(text.split())>300:
    splitters = [r"\n", r"\!", r"\?", r"\.", r","]
    # do with hierarchy
    for s in splitters:
        new_texts = []
        all_short = True
        for long_text in texts:
            if len(long_text.split())>300:
                all_short = False
                new_texts.extend([x.strip() for x in re.sub(s, _splitter, long_text).split(special_sep) if x.strip()!=''])
            else:
                new_texts.append(long_text)
        if all_short:
            # early stopping
            break
        texts = new_texts.copy()
    return texts

def pii_detection_pipeline(sample: str, split_to_short = False, both_short_and_long = False) -> bool:
    samples = None
    if split_to_short or both_short_and_long:
        samples = split_long_text(sample)
    if both_short_and_long:
        if len(samples)>1:
            samples+=[sample]
    
    if not samples:
        samples = [sample]
    
    answers = []
    for s in samples:
        doc = nlp_eng(s)
        answers+=[{"text":ent.text, "start_index":ent.start_char, "end_index":ent.end_char,"ner_label":ent.label_} for ent in doc.ents]
    return answers


