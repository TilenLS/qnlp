import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import datetime
import os
import sys

from discopro.grammar import tensor
from discopro.anaphora import connect_anaphora_on_top
from lambeq import BobcatParser, NumpyModel, AtomicType, Rewriter, Dataset, QuantumTrainer, SPSAOptimizer , AtomicType, IQPAnsatz, RemoveCupsRewriter, UnifyCodomainRewriter, BinaryCrossEntropyLoss
from lambeq.backend.grammar import Spider, Ty
from lambeq.backend.quantum import Box, qubit, SelfConjugate, Ry, Diagram
from contextuality.model import Model, Scenario, CyclicScenario

remove_cups = RemoveCupsRewriter()

parser = BobcatParser()
rewriter = Rewriter(['auxiliary',
                     'connector',
                     'coordination',
                     'determiner',
                     'object_rel_pronoun',
                     'subject_rel_pronoun',
                     'postadverb',
                     'preadverb',
                     'prepositional_phrase'])

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE 

ansatz = IQPAnsatz({N: 1, S: 1, P:1}, n_layers=1, n_single_qubit_params=3) 

def sent2dig(sentence1: str, sentence2: str, pro: str, ref: str, mode='default'):
    diagram1 = parser.sentence2diagram(sentence1)
    diagram2 = parser.sentence2diagram(sentence2)
    diagram = tensor(diagram1,diagram2)
    
    if mode == 'spider':
        diagram = diagram >> Spider(S, 2, 1)
    elif mode == 'box':
        merger = UnifyCodomainRewriter(Ty('s'))
        diagram = merger(diagram)
        
    pro_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == pro.casefold())
    ref_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == ref.casefold())
    final_diagram = connect_anaphora_on_top(diagram, pro_box_idx, ref_box_idx)
    rewritten_diagram = rewriter(remove_cups(final_diagram)).normal_form()
    return rewritten_diagram

def gen_labels(path: str, frac: int, verbose=False, mode='default'):
    df = pd.read_csv(path, index_col=0)
    df = df.sample(frac=frac)
    
    if not os.path.exists(os.getcwd()+'/err_logs'):
        os.mkdir(os.getcwd()+'/err_logs')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    f = open("err_logs/log_"+path.split('/')[-1].split('.')[-2]+'_'+timestamp+".txt",'w')
    
    circuits, labels, diagrams = [],[],[]
    for i, row in tqdm(df.iterrows(), total=len(df), position=0, leave=True):
        col = random.choice(['referent', 'wrong_referent'])
        sent1, sent2, pro, ref = row[['sentence1', 'sentence2', 'pronoun', col]]
        
        label = [[0.25, 0.25],[0.25, 0.25]]
        if mode == 'spider' or mode == 'box':
            label = [0, 1] if col == 'referent' else [1,0]

        try:
            diagram = sent2dig(sent1.strip(), sent2.strip(), pro.strip(), ref.strip(), mode=mode)
            diagrams.append(diagram)
            circ = ansatz(diagram)
            circuits.append(circ)
            labels.append(label)
        except Exception as err:
            tqdm.write(f"Error: {err}".strip(), file=f)
            if verbose:
                tqdm.write(f"Error: {err}".strip(), file=sys.stderr)
    f.close()
    
    return circuits, labels, diagrams

print("Generating diagrams and converting to circuits:")
mode = input("Choose diagram type (default, spider, box): ")
frac = input("What % of data to use? (<1)")
train_circuits, train_labels, train_diagrams = gen_labels('dataset/original_data/train.csv', mode=mode, frac=frac)
val_circuits, val_labels, val_diagrams = gen_labels('dataset/original_data/val.csv', mode=mode, frac=frac)
test_circuits, test_labels, test_diagrams = gen_labels('dataset/original_data/test.csv', mode=mode, frac=frac)

model = NumpyModel.from_diagrams(train_circuits + val_circuits + test_circuits, use_jit=False)
loss = BinaryCrossEntropyLoss(use_jax=True)
acc = lambda y_hat, y: np.sqrt(np.mean((np.array(y_hat)-np.array(y))**2)/2)

SEED = random.randint(0, 400)
BATCH_SIZE = 15
EPOCHS = 200

train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
val_dataset = Dataset(val_circuits, val_labels, shuffle=True)
test_dataset = Dataset(test_circuits, test_labels)

trainer = QuantumTrainer(model,
                         loss_function=loss,
                         optimizer=SPSAOptimizer,
                         epochs=EPOCHS,
                         optim_hyperparams={'a': 0.1, 'c': 0.06, 'A': 0.01 * EPOCHS},
                         evaluate_functions={"err": acc},
                         evaluate_on_train=True,
                         verbose='text', 
                         seed=SEED)

print("Learning parameters: "+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
trainer.fit(train_dataset, val_dataset, eval_interval=1, log_interval=1)
test_acc = acc(model(test_dataset.data), test_dataset.targets)
print('Test accuracy:', test_acc)
