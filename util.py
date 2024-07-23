from discopro.grammar import tensor
from discopro.anaphora import connect_anaphora_on_top
from lambeq import BobcatParser, AtomicType, RemoveCupsRewriter, UnifyCodomainRewriter, Rewriter, QuantumTrainer, Dataset, IQPAnsatz, NumpyModel
from lambeq.backend.grammar import Spider
from lambeq.backend.quantum import Ry, Diagram
import pandas as pd 
import numpy as np
from tqdm import tqdm
import datetime, os, sys, pickle, random
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

def sent2dig(sentence1: str, sentence2: str, pro: str, ref: str, join=None, cut=False):
    diagram1 = parser.sentence2diagram(sentence1)
    diagram2 = parser.sentence2diagram(sentence2)
    diagram = tensor(diagram1, diagram2)
    
    if join == 'spider':
        diagram = diagram >> Spider(S, 2, 1)
    elif join == 'box':
        merger = UnifyCodomainRewriter(S)
        diagram = merger(diagram)

    if not cut:
        pro_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == pro.casefold())
        ref_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == ref.casefold())
        diagram = connect_anaphora_on_top(diagram, pro_box_idx, ref_box_idx)
        
    diagram = rewriter(remove_cups(diagram)).normal_form()
    return diagram

def gen_labels(path: str, verbose=False, frac=1, join=None, cut=False):
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
        if join == 'spider' or join == 'box':
            label = [0, 1] if col == 'referent' else [1,0]

        try:
            diagram = sent2dig(sent1.strip(), sent2.strip(), pro.strip(), ref.strip(), join=join, cut=cut)
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

def train(trainer: QuantumTrainer, EPOCH_ARR: [int], BATCH_ARR: [int], SEED_N: int, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
    SEEDS = random.sample(range(1000), SEED_N)
    trainer.verbose = 'supress'
    model = trainer.model
    
    print("%0s %23s %7s %7s  %12s" % ("Time","Epochs","Batch","Seed","Accuracy"))
    for EPOCHS in EPOCH_ARR:
        for BATCH_SIZE in BATCH_ARR:
            for SEED in SEEDS:
                trainer.epochs = EPOCHS
                trainer.optim_hyperparams = {'a': 0.1, 'c': 0.06, 'A': 0.01 * EPOCHS}
                train_dataset.batch_size = BATCH_SIZE
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
                print("%0s %8s %7s %7s" % (time, EPOCHS, BATCH_SIZE, SEED), end='')
                trainer.fit(train_dataset, val_dataset, eval_interval=1, log_interval=1)
                test_acc = acc(model(test_dataset.data), test_dataset.targets)
                print("%14s" % (round(test_acc, 6)))

class data_loader:
    def __init__(self, scenario: Scenario, model_path: str=None):
        self.scenario = scenario # Measurement scenario modelling the schema

        # Data
        self.data = pd.DataFrame(columns=["Sentence", "CF", "SF", "CbD", "DI", "Distribution"])
        self.diagrams = []
        self.sentences = []
            
        if model_path: # NumpyModel with learnt parameters of ansatz circuits
            self.model = NumpyModel.from_checkpoint(model_path)
            self.model.initialise_weights()
        else:
            self.model = None

        # Measurement basis used in max violation CHSH experiment with their matrix representations
        self.bases = {'a':Ry(0), 'A':Ry(np.pi/4), 'b':Ry(np.pi/8), 'B':Ry(3*np.pi/8)}
        self.pairs = {'ab': np.kron(Ry(0).array, Ry(np.pi/8).array),
                      'aB': np.kron(Ry(0).array, Ry(3*np.pi/8).array),
                      'Ab': np.kron(Ry(np.pi/4).array, Ry(np.pi/8).array),
                      'AB': np.kron(Ry(np.pi/4).array, Ry(3*np.pi/8).array)}

    def load_file(self, path: str) -> None | pd.DataFrame | zip:
        if not path:
            return
        elif os.path.splitext(path)[-1] == '.csv':
            return pd.read_csv(path)
        elif os.path.splitext(path)[-1] == '.pkl':
            file = open(path, 'rb')
            data =  pickle.load(file)
            file.close()
            return data
        else:
            print("Provided file doesn't match a supported type.")
            return

    def update_model(self, path: str):
        self.model = NumpyModel.from_checkpoint(path)
        self.model.initialise_weights()

    def load_model(self, path: str, variant: str=None) -> None:
        self.model = NumpyModel.from_checkpoint(model_path)

    def get_data(self, path: str) -> None:
        if not path:
            return
        self.data = self.load_file(path)

    def get_diagrams(self, path: str, cut=True, ref_type='referent') -> None:
        if not path:
            return
        
        schema_data = self.load_file(path)
        if os.path.splitext(path)[-1] == '.pkl':
            self.sentences, self.diagrams = zip(*schema_data)
            self.sentences = list(self.sentences)
            self.diagrams = list(self.diagrams)
            return
            
        for _, row in tqdm(schema_data.iterrows(), total=len(schema_data)):
            try:
                s1, s2, pro, ref = row[['sentence1','sentence2','pronoun',ref_type]]
                self.diagrams.append(ansatz(sent2dig(s1, s2, pro, ref, cut=cut)))
                self.sentences.append(s1 + '. ' + s2 + '.')
            except Exception as err:
                tqdm.write(f"Error: {err}".strip(), file=sys.stderr)
        f = open('dataset/sent_circ_pairs'+'_'+str(len(self.diagrams))+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")+'.pkl', 'wb')
        pickle.dump(list(zip(self.sentences, self.diagrams)), f)
        f.close()

    def get_emp_model(self, diag: Diagram) -> Model:
        diag_ab = diag.apply_gate(self.bases['a'],0).apply_gate(self.bases['b'],1)
        diag_aB = diag.apply_gate(self.bases['a'],0).apply_gate(self.bases['B'],1)
        diag_Ab = diag.apply_gate(self.bases['A'],0).apply_gate(self.bases['b'],1)
        diag_AB = diag.apply_gate(self.bases['A'],0).apply_gate(self.bases['B'],1)

        pr_dist = self.model.get_diagram_output([diag_ab, diag_aB, diag_Ab, diag_AB])
        pr_dist = np.reshape(pr_dist, (4,4))
        return Model(self.scenario, pr_dist)
    
    def gen_data(self) -> None:
        data_dict = {'Sentence':[], 'CF':[], 'SF':[], 'CbD':[], 'DI':[], 'Distribution': []}
        for diagram, sentence in tqdm(zip(self.diagrams, self.sentences), total=len(self.diagrams)):
            try:
                cur_emp_model = self.get_emp_model(diagram)
                data_dict['CF'].append(cur_emp_model.signalling_fraction())
                data_dict['SF'].append(cur_emp_model.contextual_fraction())
                data_dict['CbD'].append(cur_emp_model.CbD_measure())
                data_dict['DI'].append(cur_emp_model.CbD_direct_influence())
                data_dict['Distribution'].append(cur_emp_model._distributions)
                data_dict['Sentence'].append(sentence)
            except Exception as err:
                tqdm.write(f"Error: {err}".strip(), file=sys.stderr)
        self.data = pd.DataFrame(data_dict)
        self.data.to_pickle('dataset/scenario442_' + str(len(self.diagrams)) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + '.csv')