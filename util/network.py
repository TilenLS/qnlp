from lambeq import IQPAnsatz, Sim14Ansatz, Sim15Ansatz
from discopro.grammar import tensor
from discopro.anaphora import connect_anaphora_on_top
from lambeq import BobcatParser, AtomicType, RemoveCupsRewriter, UnifyCodomainRewriter, Rewriter
from lambeq.backend.grammar import Spider

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE

remove_cups = RemoveCupsRewriter()
parser = BobcatParser()
merger = UnifyCodomainRewriter(S)
rewriter = Rewriter(['curry'])

def get_ansatz(kind="iqp", ob_map={N: 1, S: 1, P:1}, nl=1, nsqp=3):
    if kind=='iqp':
        return IQPAnsatz(ob_map, n_layers=nl, n_single_qubit_params=nsqp)
    elif kind=='sim14':
        return Sim14Ansatz(ob_map, n_layers=nl, n_single_qubit_params=nsqp)
    elif kind=='sim15':
        return Sim15Ansatz(ob_map, n_layers=nl, n_single_qubit_params=nsqp)

def sent2dig(sentence1: str, sentence2: str, pro: str, ref: str, join='spider', con_ref=True):
    diagram1 = parser.sentence2diagram(sentence1)
    diagram2 = parser.sentence2diagram(sentence2)
    diagram = tensor(diagram1, diagram2)
    
    if join == 'spider':
        diagram = diagram >> Spider(S, 2, 1)
    elif join == 'box':
        diagram = merger(diagram)

    if ref:
        pro_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == pro.casefold())
        ref_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == ref.casefold())
        diagram = connect_anaphora_on_top(diagram, pro_box_idx, ref_box_idx)
        
    diagram = rewriter(remove_cups(diagram)).normal_form()
    return diagram