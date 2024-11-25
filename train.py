import sys
from lambeq import Dataset
from util/data_processing import read_pkl
from util/model import train_eval, optimise_model

BATCH_SIZE = 10
EPOCHS = 300

train_circuits, train_labels, train_diagrams, _ = zip(*read_pkl('data/sim14_data/train.pkl'))
val_circuits, val_labels, val_diagrams, _ = zip(*read_pkl('data/sim14_data/val.pkl'))
test_circuits, test_labels, test_diagrams, _ = zip(*read_pkl('data/sim14_data/test.pkl'))

total_len = len(train_labels) + len(val_labels) + len(test_labels)
print("==============(SUMMARY)==============", file=sys.stderr)
print(f"Using batch size of [{BATCH_SIZE}] and training for [{EPOCHS}] epochs.")
print(f"Training size: {len(train_labels)} ({round(len(train_labels)/total_len,3)})", file=sys.stderr)
print(f"Validation size: {len(val_labels)} ({round(len(val_labels)/total_len,3)})", file=sys.stderr)
print(f"Test size: {len(test_labels)} ({round(len(test_labels)/total_len,3)})", file=sys.stderr)
print("=====================================", file=sys.stderr)

train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
val_dataset = Dataset(val_circuits, val_labels, shuffle=True)
test_dataset = Dataset(test_circuits, test_labels)
dataset = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset} 

train_eval(params={}, dataset=dataset, epochs=EPOCHS, verbose=True)


