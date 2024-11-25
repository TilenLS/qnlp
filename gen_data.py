from util/data_processing get_ansatz, gen_data

ansatz = get_ansatz(kind="iqp", ob_map={N: 1, S: 1, P:1}, nl=1, nsqp=3)

path_train = "data/og_data/train.csv"
path_val = "data/og_data/val.csv"
path_test = "data/og_data/test.csv"

gen_data(path_train, 'train_iqp', ansatz)
gen_data(path_val, 'val_iqp', ansatz)
gen_data(path_test, 'test_iqp', ansatz)
