from discopro.grammar import tensor
from discopro.anaphora import connect_anaphora_on_top
from lambeq import BobcatParser, AtomicType, RemoveCupsRewriter, UnifyCodomainRewriter, Rewriter, QuantumTrainer, Dataset, IQPAnsatz, NumpyModel
from lambeq.backend.grammar import Spider
from lambeq.backend.quantum import Ry, Diagram, Box, qubit
import pandas as pd 
import numpy as np
from tqdm import tqdm
import datetime, os, sys, pickle, random
from contextuality.model import Model, Scenario, CyclicScenario
from funcs import state2dense, partial_trace, calc_vne, convert_dist, gen_basis, rand_state, get_onb, calc_eoe, log_neg
import tensornetwork as tn
import matplotlib.pyplot as plt
import seaborn as sns

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
        
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        label = [[0.25, 0.25],[0.25, 0.25]]
        if sent_model == 'spider' or sent_model == 'box':
            label = [0, 1] if col == 'referent' else [1,0]
        else:
            label = [[0,0],[0,1]] if col == 'referent' else [[1,0],[0,0]]
=======
        if sent_model == 'spider' or sent_model == 'box':
            label = [0, 1] if col == 'referent' else [1,0]
        else:
            label = [[0, 0],[0, 1]] if col == 'referent' else [[1,0],[0,0]]
>>>>>>> Stashed changes
=======
        if sent_model == 'spider' or sent_model == 'box':
            label = [0, 1] if col == 'referent' else [1,0]
        else:
            label = [[0, 0],[0, 1]] if col == 'referent' else [[1,0],[0,0]]
>>>>>>> Stashed changes
=======
        if sent_model == 'spider' or sent_model == 'box':
            label = [0, 1] if col == 'referent' else [1,0]
        else:
            label = [[0, 0],[0, 1]] if col == 'referent' else [[1,0],[0,0]]
>>>>>>> Stashed changes

        try:
            diagram = sent2dig(sent1.strip(), sent2.strip(), pro.strip(), ref.strip(), sent_model=sent_model, con_ref=con_ref)
            diagrams.append(diagram)
            circuits.append(ansatz(diagram))
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
        self.data = pd.DataFrame(columns=["Sentence", "CF", "SF", "CbD", "DI", "Entropy", "Log Neg", "State", "Table"])
        self.circuits = []
        self.sentences = []
            
        if model_path:
            self.model = QModel.from_checkpoint(model_path)
            self.model.initialise_weights()

        # Measurement basis used in max violation CHSH experiment with their matrix representations
        self.onbs = (gen_basis(np.pi/2, np.pi/8)[0], gen_basis(np.pi/2, 5*np.pi/8)[0])
        self.observables = {'A': (get_observable('a1', 0, 0), get_observable('a2', np.pi/4, 0)),
                            'B': (get_observable('b1', np.pi/8, 0), get_observable('b2', 5*np.pi/8, 0))}

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

    def load_basis(self, new_obs):
        self.observables = new_obs

    def load_circuits(self, path: str, con_ref=True, ref_type='referent', save=True, frac=1) -> None:
        if os.path.splitext(path)[-1] == '.pkl':
            file = open(path, 'rb')
            schema_data =  pickle.load(file)
            file.close()
            self.sentences, self.circuits = zip(*schema_data)

            self.sentences = list(self.sentences)
            self.circuits = list(self.circuits)
            self.sentences = self.sentences[:round(frac*len(self.sentences))]
            self.circuits = self.circuits[:round(frac*len(self.circuits))]
            return
            
        schem_data = self.read_df(path)
        for _, row in tqdm(schema_data.iterrows(), total=len(schema_data)):
            try:
                s1, s2, pro, ref = row[['sentence1','sentence2','pronoun',ref_type]]
                self.circuits.append(ansatz(sent2dig(s1, s2, pro, ref, con_ref=con_ref)))
                self.sentences.append(s1 + '. ' + s2 + '.')
            except Exception as err:
                tqdm.write(f"Error: {err}".strip(), file=sys.stderr)
                
        if save:
            f = open('dataset/sent_circ_pairs'+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")+'.pkl', 'wb')
            pickle.dump(list(zip(self.sentences, self.circuits)), f)
            f.close()

    def get_contexts(self, circuit: Diagram) -> [Diagram]:
        contexts = [circuit]*4

        i = 0
        for obs1 in self.observables['A']:
            for obs2 in self.observables['B']:
                contexts[i] = contexts[i].apply_gate(obs1, 0)
                contexts[i] = contexts[i].apply_gate(obs2, 1)
                i += 1
            
        return contexts

    def get_dist(self, state):
        prs1 = abs(np.kron(self.onb[0], self.onb[0]) @ state)**2
        prs2 = abs(np.kron(self.onb[0], self.onb[1]) @ state)**2
        prs3 = abs(np.kron(self.onb[1], self.onb[0]) @ state)**2
        prs4 = abs(np.kron(self.onb[1], self.onb[1]) @ state)**2
        return np.array([prs1, prs2, prs3, prs4])

    def get_emp_model(self, contexts: [Diagram]) -> Model:
        # Measurement contexts ordered to coincide with the cyclic measurement scenario
        pr_dist = self.model.get_diagram_output(contexts)
        pr_dist = np.reshape(pr_dist, (4,4))
        return Model(self.scenario, convert_dist(pr_dist))
    
    def gen_data(self, save=True, tol=6) -> None:
        data_dict = {'Sentence':[], 'CF':[], 'SF':[], 'DI':[], 'CbD':[], 'Entropy':[], 'LogNeg':[], 'State':[], 'Distribution':[]}
        for circuit, sentence in tqdm(zip(self.circuits, self.sentences), total=len(self.circuits)):
            try:
                emp_model = self.get_emp_model(self.get_contexts(circuit))
                cf = round(emp_model.contextual_fraction(), tol)
                sf = round(emp_model.signalling_fraction(), tol)
                di = round(emp_model.CbD_direct_influence(), tol)
                cbd = round(emp_model.CbD_measure(), tol)
                dist = convert_dist(emp_model._distributions, True)
                
                state = self.model.get_output_state([circuit])[0]
                dense_mat = state2dense(state)
                eoe = calc_eoe(dense_mat)
                lneg = log_neg(dense_mat)

            
                data_dict['Sentence'].append(sentence)
                data_dict['CF'].append(cf)
                data_dict['SF'].append(sf)
                data_dict['DI'].append(di)
                data_dict['CbD'].append(cbd)
                data_dict['Entropy'].append(eoe)
                data_dict['LogNeg'].append(lneg)
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
    return (theta_arr, phi_arr, np.array(states, dtype=np.complex128))

def gen_states(num, rand=False, ghz=False):
    states = []
    if rand:
        for n in range(num):
            states.append(rand_state(ghz=ghz))
    else:
        lim = round(num**(1/4))
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

def gen_contexts(num=10000):
    contexts = []
    num = round(num**0.5)
    phi_arr = np.linspace(0, np.pi, num)
    for phi1 in phi_arr:
        for phi2 in phi_arr:
            onb1 = gen_basis(np.pi/2, phi1)
            onb2 = gen_basis(np.pi/2, phi2)
            context = {'ab': np.kron(onb1[0], onb1[0]), 
                       'aB': np.kron(onb1[0], onb2[0]), 
                       'Ab': np.kron(onb2[0], onb1[0]), 
                       'AB': np.kron(onb2[0], onb2[0])}
            contexts.append(context)
    return np.array(contexts)

def get_observable(name="O", theta=0, phi=0):
    return Box(name=name, dom=qubit, cod=qubit, data=get_onb(theta, phi))

def plot_scatter(fig, x, y, z):
    cmap = plt.get_cmap('viridis_r')
    cmap.set_under('red')
    scat = fig.scatter(x=x, y=y, c=z, cmap=cmap)
    fig.set_alpha(0.5)
    plt.colorbar(scat, extend='min')
    return fig

def plot_dist(fig, data_arr, labels, title, cumulative=False):
    if len(data_arr) != len(labels):
        return "Number of datasets does not match number of labels"
    cmap = plt.get_cmap('turbo', len(data_arr))
    for i in range(len(data_arr)):
        sns.kdeplot(data=data_arr[i], color=cmap(i), label=labels[i], fill=True, ax=fig, cut=0, cumulative=cumulative, 
                    common_norm=True, common_grid=True, log_scale=False)
    fig.legend()
    fig.set_title(title)
    return fig

def plot_heatmap(fig, x, y, z):
    X,Y = np.meshgrid(x,y)
    Z=z.reshape(len(x),len(y))
    Z=np.transpose(Z)
    hmap = fig.imshow(Z , cmap = 'jet' , interpolation = 'gaussian' , 
           origin='lower', aspect='equal',  extent = [min(x), max(x), min(y), max(y)])
    plt.colorbar(hmap, extend='min')
    return fig
