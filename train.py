import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from discopro.grammar import tensor
from lambeq import BobcatParser, NumpyModel, AtomicType, Rewriter, Dataset, QuantumTrainer, SPSAOptimizer , AtomicType, IQPAnsatz, RemoveCupsRewriter
from lambeq.backend.grammar import Ty
from tqdm import tqdm
import random
import datetime
from discopro.anaphora import connect_anaphora_on_top
import sys

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

def generate_diagram(diagram, pro, ref):

    pro_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == pro.casefold())
    ref_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == ref.casefold())
    final_diagram = connect_anaphora_on_top(diagram, pro_box_idx, ref_box_idx)
    rewritten_diagram = rewriter(remove_cups(final_diagram)).normal_form()
    return rewritten_diagram

def sent2dig(sentence1: str, sentence2: str, pro: str, ref: str):
    diagram1 = parser.sentence2diagram(sentence1)
    diagram2 = parser.sentence2diagram(sentence2)
    diagram = tensor(diagram1,diagram2)
    diagram = generate_diagram(diagram, pro, ref)
    return diagram

def gen_labels(df: pd.DataFrame):
    circuits, labels, diagrams = [],[],[]
    #selected_cols = [random.choice(['referent', 'wrong_referent']) for i in range(len(df))]
    for i, row in tqdm(df.iterrows(), total=len(df)):
        #ref = row[selected_cols[i]]
        # label = [[0.25, 0.25],[0.25, 0.25]] if selected_cols[i] == 'referent' else [[0.25, 0.25],[0.25, 0.25]]
        label = [[0.25, 0.25],[0.25, 0.25]]
        sent1, sent2, pro, ref = row[['sentence1', 'sentence2', 'pronoun', 'referent']]

        try:
            diagram = sent2dig(sent1.strip(), sent2.strip(), pro.strip(), ref.strip())
            diagrams.append(diagram)
            circ = ansatz(diagram)
            circuits.append(circ)
            labels.append(label)
        except Exception as e:
            tqdm.write(f"Error: {e}".strip(), file=sys.stderr)
    return circuits, labels, diagrams

df_train = pd.read_csv('dataset/original_data/train.csv', index_col=0)
df_val = pd.read_csv('dataset/original_data/val.csv', index_col=0)
df_test = pd.read_csv('dataset/original_data/test.csv', index_col=0)

print("Generating diagrams and converting to circuits:")
train_circuits, train_labels, train_diagrams = gen_labels(df_train[:10])
val_circuits, val_labels, val_diagrams = gen_labels(df_val[:10])
test_circuits, test_labels, test_diagrams = gen_labels(df_test[:10])

all_circuits = train_circuits + val_circuits + test_circuits
model = NumpyModel.from_diagrams(all_circuits, use_jit=True)
loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
acc = lambda y_hat, y: np.sum(np.round(y_hat) == np.array(y)) / len(y) / 2  # half due to double-counting
eval_metrics = {"acc": acc}

def main(EPOCHS: int, SEED: int, BATCH_SIZE: int) -> None:

    trainer = QuantumTrainer(
        model,
        loss_function=loss,
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.1, 'c': 0.06, 'A': 0.01 * EPOCHS},
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=SEED)

    train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
    val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

    now = datetime.datetime.now()
    t = now.strftime("%Y-%m-%d_%H_%M_%S")
    print(t)
    trainer.fit(train_dataset, val_dataset, eval_interval=1, log_interval=1)
    test_acc = acc(model(test_circuits), test_labels)
    print('Test accuracy:', test_acc)

print("Learning circuit parameters:")
main(100, random.randrange(0,400), 10)