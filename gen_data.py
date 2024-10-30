from lambeq import BobcatParser, AtomicType, RemoveCupsRewriter, Rewriter, Sim14Ansatz, IQPAnsatz
from discopro.grammar import tensor
from discopro.anaphora import connect_anaphora_on_top
from util import sent2dig
import pandas as pd
from tqdm import tqdm
import sys, random, pickle

remove_cups = RemoveCupsRewriter()

parser = BobcatParser()
rewriter = Rewriter(['curry'])

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE

ansatz = Sim14Ansatz({N: 1, S: 1, P:1}, n_layers=1, n_single_qubit_params=3)

path = "data/og_data"

def sent2dig(sentence1: str, sentence2: str, pro: str, ref: str):
    diagram1 = parser.sentence2diagram(sentence1)
    diagram2 = parser.sentence2diagram(sentence2)
    diagram = tensor(diagram1, diagram2)
    diagram = diagram >> Spider(S, 2, 1)

    pro_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == pro.casefold())
    ref_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == ref.casefold())
    diagram = connect_anaphora_on_top(diagram, pro_box_idx, ref_box_idx)
        
    diagram = rewriter(remove_cups(diagram)).normal_form()
    return diagram

def gen_data(dir_path, file_name):
    df = pd.read_csv(path + '/' + file_name + '.csv', index_col=0)
    df = df.sample(frac=1)
    
    circuits, labels, diagrams = [],[],[]
    
    for i, row in tqdm(df.iterrows(), total=len(df), position=0, leave=True):
        col = random.choice(['referent', 'wrong_referent'])
        sent1, sent2, pro, ref = row[['sentence1', 'sentence2', 'pronoun', col]]
        label = [0, 1] if col == 'referent' else [1,0]
        try:
            diagram = sent2dig(sent1.strip(), sent2.strip(), pro.strip(), ref.strip())
            diagrams.append(diagram)
            circuits.append(ansatz(diagram))
            labels.append(label)
        except Exception as err:
            tqdm.write(f"Error: {err}".strip(), file=sys.stderr)
    
    f = open('data/sim14_data/'+file_name+'.pkl', 'wb')
    pickle.dump(list(zip(circuits, labels, diagrams)), f)
    f.close()

gen_data(path, 'train')
gen_data(path, 'val')
gen_data(path, 'test')
