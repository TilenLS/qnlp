from .quantum import gen_rot, gen_basis, gen_obs 
import numpy as np

bell_state = np.sqrt([1/2,0,0,1/2], dtype=np.complex128)
ghz_state = np.sqrt([1/2,0,0,0,0,0,0,1/2], dtype=np.complex128)

onb = (gen_rot(np.pi/2, np.pi/8), gen_rot(np.pi/2, 5*np.pi/8))

contexts_1 = {'ab': np.kron(onb[0], onb[0]), 
              'aB': np.kron(onb[0], onb[1]),  
              'Ab': np.kron(onb[1], onb[0]),  
              'AB': np.kron(onb[1], onb[1])}
observables_1 = {'A': (gen_obs('a1', onb[0]), gen_obs('a2', onb[1])), 
                 'B': (gen_obs('b1', onb[0]), gen_obs('b2', onb[1]))}

contexts_2 = {'ab': np.kron(gen_rot(), gen_rot(np.pi/8)), 
              'aB': np.kron(gen_rot(), gen_rot(3*np.pi/8)),  
              'Ab': np.kron(gen_rot(np.pi/4), gen_rot(np.pi/8)),  
              'AB': np.kron(gen_rot(np.pi/4), gen_rot(3*np.pi/8))}
observables_2 = {'A': (gen_obs('a1'), gen_obs('a2', theta=np.pi/4)), 
                 'B': (gen_obs('b1', theta=np.pi/8), gen_obs('b2', theta=3*np.pi/8))}

subj_obj = [('heart', 'blood'), ('sailors', 'boats'), ('students', 'books'), ('knife', 'fence'),
                  ('storm', 'flight'), ('bee', 'flower'), ('birds', 'seeds'), ('police', 'criminals'), 
                  ('people', 'government'), ('sniper', 'terrorist')]

model_path = {'d_ref': 'runs/disjoint_ref_130E/best_model.lt', 
              'd_nref': 'runs/disjoint_noref_140E/best_model.lt',
              's_ref': 'runs/spider_ref_200E/best_model.lt', 
              's_nref': 'runs/spider_noref_50E/best_model.lt', 
              'sim14': 'runs/sim14_model/best_model.lt', 
              'sim15': 'runs/sim15_model/best_model.lt'}

diagram_path = {'rref': 'dataset/diagrams/diags_right_ref.pkl',
                'wref': 'dataset/diagrams/diags_wrong_ref.pkl',
                'nref': 'dataset/diagrams/diags_no_ref.pkl', 
                'mixed_ref': 'data/diagrams/_diags_mixed_ref.pkl'}

data_path = {'d_rref': 'data/results/open-wire_right_ref.pkl',
             'd_wref': 'data/results/open-wire_wrong_ref.pkl',
             'd_nref': 'data/results/open-write_no_ref.pkl',
             's_rref': 'data/results/spider_right_ref.pkl',
             's_wref': 'data/results/spider_wrong_ref.pkl',
             's_nref': 'data/results/spider_no_ref.pkl', 
             'sim14': 'data/sim14_data/sim14_results.pkl', 
             'sim15': 'data/sim15_data/sim15_results.pkl'}