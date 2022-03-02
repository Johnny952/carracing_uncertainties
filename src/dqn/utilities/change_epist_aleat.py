import numpy as np

def change_uncerts(path, fix_path):
    with open(path, "r") as f:
        with open(fix_path, "a+") as fix_f:
            for row in f:
                data = np.array(row[:-1].split(",")).astype(np.float32)
                epoch = data[0]
                val_idx = data[1]
                reward = data[2]
                sigma = data[3]
                l = len(data) - 4
                epist = data[4 : l // 2 + 4]
                aleat = data[l // 2 + 4 :]

                np.savetxt(fix_f, np.concatenate(([epoch], [val_idx], [reward], [sigma], aleat, epist)).reshape(1, -1), delimiter=',')

if __name__ == "__main__":
    file = 'uncertainties/train/ddqn_dropout_481dd29d-3ecc-48aa-9249-31896c98036f.txt'
    fix_file = 'uncertainties/train/fix_ddqn_dropout_481dd29d-3ecc-48aa-9249-31896c98036f.txt'
    
    change_uncerts(file, fix_file)

