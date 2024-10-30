from lambeq import BobcatParser, AtomicType, RemoveCupsRewriter, Rewriter, Sim14Ansatz, IQPAnsatz
from util import sent2dig
import pandas as pd
from tqdm import tqdm

remove_cups = RemoveCupsRewriter()

parser = BobcatParser()
rewriter = Rewriter(['curry'])

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE

ansatz = Sim14Ansatz({N: 1, S: 1, P:1}, n_layers=1, n_single_qubit_params=3)

path = ""

df = pd.read_csv(path, index_col=0)
df = df.sample(frac=frac)

circuits, labels, diagrams = [],[],[]

for i, row in tqdm(df.iterrows(), total=len(df), position=0, leave=True):
    col = random.choice(['referent', 'wrong_referent'])
    sent1, sent2, pro, ref = row[['sentence1', 'sentence2', 'pronoun', col]]
    label = [0, 1] if col == 'referent' else [1,0]
    try:
        diagram = sent2dig(sent1.strip(), sent2.strip(), pro.strip(), ref.strip(), sent_model='spider', con_ref=True)
        diagrams.append(diagram)
        circuits.append(ansatz(diagram))
        labels.append(label)
    except Exception as err:
        tqdm.write(f"Error: {err}".strip(), file=sys.stderr)

f = open('dataset/data_cld_'+datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")+'.pkl', 'wb')
pickle.dump(list(zip(circuits, labels, diagrams)), f)
f.close()