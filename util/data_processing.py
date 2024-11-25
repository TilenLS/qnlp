import pandas as pd 
import numpy as np
import tensornetwork as tn
from lambeq import NumpyModel
from tqdm import tqdm
from lambeq.backend.quantum import Diagram
import datetime, os, pickle, random, sys
from contextuality.model import Model, Scenario
from .constants import contexts_1, observables_2
from .quantum import state2dense, ent_ent, log_neg
from .network import sent2dig, get_ansatz

def read_pkl(path: str):
    file = open(path, 'rb')
    data =  pickle.load(file)
    file.close()
    return data

def write_pkl(path: str, data):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()

def gen_data(path: str, name: str, ansatz = get_ansatz(), join='spider', con_ref=True, save=True):
    df = pd.read_csv(path, index_col=0)
    
    circuits, labels, diagrams, sentences = [],[],[], []
    for i, row in tqdm(df.iterrows(), total=len(df), position=0, leave=True):
        col = random.choice(['referent', 'wrong_referent'])
        sent1, sent2, pro, ref = row[['sentence1', 'sentence2', 'pronoun', col]]
        
        if join:
            label = [0, 1] if col == 'referent' else [1,0]
        else:
            label = [[0, 0],[0, 1]] if col == 'referent' else [[1,0],[0,0]]

        try:
            diagram = sent2dig(sent1.strip(), sent2.strip(), pro.strip(), ref.strip(), join=join, con_ref=con_ref)
            diagrams.append(diagram)
            circuits.append(ansatz(diagram))
            labels.append(label)
            sentences.append(sent1 + '. ' + sent2 + '.')
        except Exception as err:
            tqdm.write(f"Error: {err}".strip(), file=sys.stder)
            
    if save:
        write_pkl('dataset/'+name+'.pkl', list(zip(circuits, labels, diagrams, sentences)))
        return
        
    return circuits, labels, diagrams, sentences

def conv_dist(pr_dist: np.ndarray, is_cyc=False) -> np.ndarray:
    if is_cyc:
        # Converts a cyclic distribution to a standard one 
        new_dist = np.zeros_like(pr_dist)
        new_dist[0] = pr_dist[0]
        new_dist[1] = pr_dist[3][[0,2,1,3]]
        new_dist[2] = pr_dist[1][[0,2,1,3]]
        new_dist[3] = pr_dist[2]
    else:
        # Converts a standard distribution to a cyclic one
        new_dist = np.zeros_like(pr_dist)
        new_dist[0] = pr_dist[0]
        new_dist[1] = pr_dist[2][[0,2,1,3]]
        new_dist[2] = pr_dist[3]
        new_dist[3] = pr_dist[1][[0,2,1,3]]
    return new_dist

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

class data_loader:
    def __init__(self, scenario: Scenario, model_path: str=None):
        self.scenario = scenario # Measurement scenario modelling the schema
        if model_path:
            self.load_model(model_path)
        self.data = pd.DataFrame(columns=["Sentence", "CF", "SF", "CbD", "DI", "Entropy", "LogNeg", "State", "Table"])
        self.circuits = []
        self.labels = []
        self.diagrams = []
        self.sentences = []
        # Measurement basis used in max violation CHSH experiment with their matrix representations
        self.contexts = contexts_1
        self.observables = observables_2

    def read_df(self, path: str):
        if os.path.splitext(path)[-1] == '.csv':
            return pd.read_csv(path)
        elif os.path.splitext(path)[-1] == '.pkl':
            return pd.read_pickle(path)

    def load_model(self, path: str, variant: str=None) -> None:
        self.model = QModel.from_checkpoint(path)
        self.model.initialise_weights()

    def load_circuits(self, path: str) -> None:
        schema_data =  read_pkl(file)
        self.circuits, self.labels, self.diagrams, self.sentences = zip(*schema_data)

        self.circuits = list(self.circuits)
        self.labels = list(self.labels)
        self.diagrams = list(self.diagrams)
        self.sentences = list(self.sentences)

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
        prs1 = abs(self.contexts['ab'] @ state)**2
        prs2 = abs(self.contexts['aB'] @ state)**2
        prs3 = abs(self.contexts['Ab'] @ state)**2
        prs4 = abs(self.contexts['AB'] @ state)**2
        return np.array([prs1, prs2, prs3, prs4])

    def get_emp_model(self, contexts: [Diagram]) -> Model:
        # Measurement contexts ordered to coincide with the cyclic measurement scenario
        pr_dist = self.model.get_diagram_output(contexts)
        pr_dist = np.reshape(pr_dist, (4,4))
        return Model(self.scenario, conv_dist(pr_dist))
    
    def get_results(self, save=True, tol=6) -> None:
        data_dict = {'Sentence':[], 'CF':[], 'SF':[], 'DI':[], 'CbD':[], 'Entropy':[], 'LogNeg':[], 'State':[], 'Distribution':[]}
        for circuit, sentence in tqdm(zip(self.circuits, self.sentences), total=len(self.circuits)):
            try:
                emp_model = self.get_emp_model(self.get_contexts(circuit))
                cf = round(emp_model.contextual_fraction(), tol)
                sf = round(emp_model.signalling_fraction(), tol)
                di = round(emp_model.CbD_direct_influence(), tol)
                cbd = round(emp_model.CbD_measure(), tol)
                dist = conv_dist(emp_model._distributions, True)
                
                state = self.model.get_output_state([circuit])[0]
                dense_mat = state2dense(state)
                eoe = ent_ent(dense_mat)
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
            self.data.to_pickle('data/results/qinfo_'+datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")+'.pkl')

def get_sim(df_arr):
    common_sentences = [[] for _ in df_arr]
    min_i = 0
    for i in range(len(df_arr)):
        if len(df_arr[i]) < len(df_arr[min_i]):
            min_i = i
    min_df = df_arr.pop(min_i)
    for i, row in min_df.iterrows():
        res_arr = []
        for df in df_arr:
            res = df.loc[df['Sentence'] == row['Sentence']]
            if len(res) == 0:
                break
            res_arr.append(res.index[0])
        if len(res_arr) == len(df_arr):
            res_arr.insert(min_i, i)
            for j in range(len(res_arr)):
                common_sentences[j].append(res_arr[j])
    df_arr.insert(min_i, min_df)
    return [df.iloc[ints] for df, ints in zip(df_arr, common_sentences)]
