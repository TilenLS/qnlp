import numpy as np 
import random
import datetime
import sys
from lambeq import NumpyModel, Dataset, QuantumTrainer, SPSAOptimizer, BinaryCrossEntropyLoss
from util import sent2dig, gen_labels, train

frac = int(sys.argv[1])/100
join = str(sys.argv[2])
cut = bool(sys.argv[3])

print("Generating diagrams and converting to circuits:")

train_circuits, train_labels, train_diagrams = gen_labels('dataset/original_data/train.csv', 
                                                          frac=frac, join=join, cut=cut)
val_circuits, val_labels, val_diagrams = gen_labels('dataset/original_data/val.csv', 
                                                    frac=frac, join=join, cut=cut)
test_circuits, test_labels, test_diagrams = gen_labels('dataset/original_data/test.csv', 
                                                       frac=frac, join=join, cut=cut)

model = NumpyModel.from_diagrams(train_circuits + val_circuits + test_circuits, use_jit=True)
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
print("Seed: " + str(SEED))
trainer.fit(train_dataset, val_dataset, eval_interval=1, log_interval=1)
test_acc = acc(model(test_dataset.data), test_dataset.targets)
print('Test accuracy:', test_acc)
#train(trainer, [50,100,150,200], [5,10,20,30], random.sample(range(500), 5), train_dataset, val_dataset, test_dataset)
