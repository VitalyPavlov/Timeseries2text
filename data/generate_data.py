import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main():
    nsteps = 20_000
    df = pd.DataFrame(columns=['seq', 'target'])
    for step in range(nsteps):
        seq = []
        target = []
        seg_size = np.random.randint(2, 10)
        empty_space_size = np.random.randint(10, 50)
        zero_size = np.random.randint(4, 10)
        one_size = np.random.randint(3, 6)
        for _ in range(seg_size):
            empty_space = np.zeros((32, empty_space_size))
            seq.append(empty_space)    

            number = np.random.randint(2, 6)
            target.append(number)
            for _ in range(number):
                box = np.concatenate((np.zeros((32, zero_size)), np.ones((32, one_size))), axis=1)
                seq.append(box)

        seq = np.concatenate(seq, axis=1)
        seq += np.random.normal(0, 0.5, seq.shape)

        index = len(df)
        df.at[index, 'seq'] = seq.reshape(-1)
        df.at[index, 'target'] = target
    
    df['stratification'] = df['target'].apply(lambda x: len(x))

    _, X_test, _, _ = train_test_split(df.index, df.index, test_size=0.33, 
                                       random_state=42, stratify=df.stratification)
    
    df['fold'] = 'train'
    df.loc[X_test, 'fold'] = 'test'

    df.to_parquet('./data/dataset.parquet', index=False)

if __name__ == '__main__':
    main()

