import numpy as np 
import datetime, sys
from lambeq import NumpyModel, QuantumTrainer, SPSAOptimizer, BinaryCrossEntropyLoss
from ax.service.ax_client import AxClient, ObjectiveProperties

model = NumpyModel.from_diagrams(train_circuits + val_circuits + test_circuits, use_jit=False)
loss = BinaryCrossEntropyLoss(use_jax=True)
rmse = lambda y_hat, y: np.sqrt(np.mean((np.array(y_hat)-np.array(y))**2)/2)
bce = lambda y_hat, y: np.sum(np.round(y_hat) == y)/len(y)/2

def train_eval(params: dict, dataset: dict, epochs: int, verbose=True):
    trainer = QuantumTrainer(model,
                             loss_function=loss,
                             optimizer=SPSAOptimizer,
                             epochs=epochs,
                             optim_hyperparams={'a': params.get('a', 0.1), 
                                                'c': params.get('c', 0.06), 
                                                'A': params.get('A', 0.01) * epochs},
                             evaluate_functions={'rmse': rmse, 'bce': bce},
                             evaluate_on_train=True,
                             verbose='text', 
                             seed=params.get('seed', 42))
    if verbose:
        print("Learning parameters: "+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"), file=sys.stderr)
    trainer.fit(dataset.get('train'), dataset.get('val'), eval_interval=1, log_interval=1)
    test_acc = bce(model(dataset.get('test').data), dataset.get('test').targets)
    if verbose:
        print('Test accuracy:', test_acc, file=sys.stderr)
    return test_acc 

def optimise_model(n_trials: int,  epochs: int, dataset: dict)
    ax_client = AxClient
    ax_client.create_experiment(name='bayesianopt', 
                                parameters=[{'name': 'a', 'type': 'range', 'bounds': [1e-4, 1e-1], 'log_scale': True}, 
                                            {'name': 'c', 'type': 'range', 'bounds': [1e-4, 1e-1], 'log_scale': True}, 
                                            {'name': 'A', 'type': 'range', 'bounds': [1e-4, 1e-1], 'log_scale': True}, 
                                            {'name': 'seed', 'type': 'range', 'bounds': [0, 500]}], 
                                            objectives={'accuracy': ObjectiveProperties(minimize=False)})
    
    ax_client.attach_trial(parameters={'a': 0.1, 'c': 0.06, 'A': 0.01, 'seed': 42})
    baseline_parameters= ax.client.get_trial_parameters(trial_index=0)
    ax_client.complete_trial(trial_index=0, raw_data=train_eval(baseline_parameters, dataset, epochs, False))
    
    for _ in rage(n_trials):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=train_eval(parameters, dataset, epochs, False))
    
    best_parameters, metrics = ax_client.get_best_parameters()
    print("The best performing parameters are: ", file=sys.stderr)

