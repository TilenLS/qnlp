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
from funcs import state2dense, partial_trace, calc_vne, std2cyc, cyc2std
import tensornetwork as tn

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

class QModel(NumpyModel):
    def __init__(self, use_jit: bool = False) -> None:
        super().__init__(use_jit)

    def get_output_state(self, diagrams):
        diagrams = self._fast_subs(diagrams, self.weights)
        results = []
        for d in diagrams:
            assert isinstance(d, Diagram)
            result = tn.contractors.auto(*d.to_tn()).tensor
            result = np.array(result).flatten()
            result = np.sqrt(result/sum(abs(result)))
            results.append(result)
        return np.array(results)

def sent2dig(sentence1: str, sentence2: str, pro: str, ref: str, sent_model=None, con_ref=True):
    diagram1 = parser.sentence2diagram(sentence1)
    diagram2 = parser.sentence2diagram(sentence2)
    diagram = tensor(diagram1, diagram2)
    
    if sent_model == 'spider':
        diagram = diagram >> Spider(S, 2, 1)
    elif sent_model == 'box':
        merger = UnifyCodomainRewriter(S)
        diagram = merger(diagram)

    if con_ref:
        pro_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == pro.casefold())
        ref_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == ref.casefold())
        diagram = connect_anaphora_on_top(diagram, pro_box_idx, ref_box_idx)
        
    diagram = rewriter(remove_cups(diagram)).normal_form()
    return diagram

def gen_labels(path: str, verbose=False, frac=1, sent_model=None, con_ref=True):
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
            diagram = sent2dig(sent1.strip(), sent2.strip(), pro.strip(), ref.strip(), sent_model=sent_model, con_ref=con_ref)
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
        self.data = pd.DataFrame(columns=["Sentence", "CF", "SF", "CbD", "DI", "Entropy", "Distribution"])
        self.diagrams = []
        self.sentences = []
            
        # NumpyModel with learnt parameters of ansatz circuits
        if model_path:
            self.model = QModel.from_checkpoint(model_path)
            self.model.initialise_weights()

        # Measurement basis used in max violation CHSH experiment with their matrix representations
        self.observables = {'a':Ry(0), 'b':Ry(np.pi/8), 'A':Ry(np.pi/4), 'B':Ry(3*np.pi/8)}

    def read_df(self, path: str):
        if os.path.splitext(path)[-1] == '.csv':
            return pd.read_csv(path)
        elif os.path.splitext(path)[-1] == '.pkl':
            return pd.read_pickle(path)
        else:
            print("Provided file doesn't match a supported type.")
            return

    def load_model(self, path: str, variant: str=None) -> None:
        self.model = QModel.from_checkpoint(path)
        self.model.initialise_weights()

    def load_data(self, path: str) -> None:
        if not path:
            return
        self.data = self.read_df(path)

    def load_basis(self, new_basis):
        self.observables = new_basis

    def load_diagrams(self, path: str, con_ref=True, ref_type='referent', save=True, frac=1) -> None:
        if os.path.splitext(path)[-1] == '.pkl':
            file = open(path, 'rb')
            schema_data =  pickle.load(file)
            file.close()
            self.sentences, self.diagrams = zip(*schema_data)
            self.sentences = list(self.sentences)
            self.diagrams = list(self.diagrams)
            self.sentences = self.sentences[:round(frac*len(self.sentences))]
            self.diagrams = self.diagrams[:round(frac*len(self.diagrams))]
            return
            
        schem_data = self.read_df(path)
        for _, row in tqdm(schema_data.iterrows(), total=len(schema_data)):
            try:
                s1, s2, pro, ref = row[['sentence1','sentence2','pronoun',ref_type]]
                self.diagrams.append(ansatz(sent2dig(s1, s2, pro, ref, con_ref=con_ref)))
                self.sentences.append(s1 + '. ' + s2 + '.')
            except Exception as err:
                tqdm.write(f"Error: {err}".strip(), file=sys.stderr)
                
        if save:
            f = open('dataset/sent_circ_pairs'+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")+'.pkl', 'wb')
            pickle.dump(list(zip(self.sentences, self.diagrams)), f)
            f.close()

    def get_emp_model(self, circuit: Diagram) -> Model:
        # Measurement contexts ordered to coincide with the cyclic measurement scenario
        context_ab = circuit.apply_gate(self.observables['a'],0)
        context_ab = context_ab.apply_gate(self.observables['b'],1)
        
        context_aB = circuit.apply_gate(self.observables['a'],0)
        context_aB = context_aB.apply_gate(self.observables['B'],1)
        
        context_Ab = circuit.apply_gate(self.observables['A'],0)
        context_Ab = context_Ab.apply_gate(self.observables['b'],1)
        
        context_AB = circuit.apply_gate(self.observables['A'],0)
        context_AB = context_AB.apply_gate(self.observables['B'],1)
        
        pr_dist = self.model.get_diagram_output([context_ab, context_aB, context_Ab, context_AB])
        pr_dist = np.reshape(_pr_dist, (4,4))
        
        return Model(self.scenario, std2cyc(pr_dist))
    
    def gen_data(self, save=True, tol=6) -> None:
        data_dict = {'Sentence':[], 'CF':[], 'SF':[], 'DI':[], 'CbD':[], 'Entropy':[], 'State':[], 'Distribution':[]}
        for diagram, sentence in tqdm(zip(self.diagrams, self.sentences), total=len(self.diagrams)):
            try:
                emp_model = self.get_emp_model(diagram)
                cf = round(emp_model.contextual_fraction(), tol)
                sf = round(emp_model.signalling_fraction(), tol)
                cbd = round(emp_model.CbD_measure(), tol)
                di = round(emp_model.CbD_direct_influence(), tol)
                dist = cyc2std(emp_model._distributions)
                
                state = self.model.get_output_state([diagram])[0]
                rho_a, rho_b = partial_trace(state2dense(state))
                eoe = round(calc_vne(rho_a), tol)
            
                data_dict['Sentence'].append(sentence)
                data_dict['CF'].append(cf)
                data_dict['SF'].append(sf)
                data_dict['DI'].append(cbd)
                data_dict['CbD'].append(di)
                data_dict['Entropy'].append(eoe)
                data_dict['State'].append(state)
                data_dict['Distribution'].append(dist)
            except Exception as err:
                tqdm.write(f"Error: {err}".strip(), file=sys.stderr)
        self.data = pd.DataFrame(data_dict)
        if save:
            self.data.to_pickle('dataset/s442_'+datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")+'.pkl')

def gen_bloch_states(num, rand=False):
    states = []
    num = round(num**0.5)
    if rand:
        theta_arr = np.random.uniform(0, np.pi, num)
        phi_arr = np.random.uniform(0, np.pi, num)
    else:
        theta_arr = np.linspace(0, np.pi, num)
        phi_arr = np.linspace(0, np.pi, num)

    for theta in theta_arr:
        for phi in phi_arr:
            up = np.cos(theta/2)
            down = np.sin(theta/2)*(np.cos(phi)+1j*np.sin(phi))
            state = np.array([up, 0, 0, down], dtype=np.complex128)
            states.append(state)
    return np.array(states, dtype=np.complex128)

def gen_states(num, rand=False):
    states = []
    lim = round(num**(1/4))
    if rand:
        amps = np.random.uniform(size=lim)
    else:
        amps = np.linspace(0,1,lim)
    for alpha in amps:
        for beta in amps:
            for gamma in amps:
                for delta in amps:
                    _ = np.array([alpha, beta, gamma, delta])
                    if sum(_) == 0:
                        continue
                    states.append(np.sqrt(_ / sum(_)))
    return np.array(states, dtype=np.complex128)
