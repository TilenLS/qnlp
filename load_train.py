import numpy as np 
import random
import datetime
import sys
from lambeq import NumpyModel, Dataset, QuantumTrainer, SPSAOptimizer, BinaryCrossEntropyLoss
from util import sent2dig, gen_labels, train

path = ""
frac = 1
sent_model = 'open wire'
con_ref = True 
SEED = 200
BATCH_SIZE = 15
EPOCHS = 200 - (0)

printf("===========(MODEL SUMMARY)===========")
print(f"Referents: {con_ref}...")
print(f"Measurements: {sent_model}...")
print(f"Epochs: {EPOCHS}...")
print(f"Batch: {BATCH_SIZE}...")
print(f"SEED: {SEED}...")
printf("=====================================")

train_circuits, train_labels, train_diagrams = gen_labels('dataset/original_data/train.csv', frac=frac, sent_model=sent_model, con_ref=con_ref) 
val_circuits, val_labels, val_diagrams = gen_labels('dataset/original_data/val.csv', frac=frac, sent_model=sent_model, con_ref=con_ref)  
test_circuits, test_labels, test_diagrams = gen_labels('dataset/original_data/test.csv', frac=frac, sent_model=sent_model, con_ref=con_ref)  

model = NumpyModel.from_checkpoint(path, use_jit=False)

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
