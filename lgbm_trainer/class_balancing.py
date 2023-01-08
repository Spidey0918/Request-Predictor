# START: OWN CODE
import pandas as pd
import sys


def balance_data(input_data, label_ind, category_ind = 14):
    pos_data = input_data[input_data.iloc[:, label_ind] > 0]
    neg_data = input_data[input_data.iloc[:, label_ind] == 0]
    print('SystemLog: Positive instances {} Negative instances {} before balancing'.format(pos_data.shape[0],
                                                                                           neg_data.shape[0]))
    if neg_data.shape[0] < pos_data.shape[0]:
        pos_data = pos_data.sample(neg_data.shape[0], random_state=0)
    else:
        neg_data = neg_data.sample(pos_data.shape[0], random_state=0)

    # concatenate
    balanced_data = pd.concat([pos_data, neg_data]).sample(frac=1, random_state=0).reset_index(drop=True)
    print('SystemLog: Positive instances {} Negative instances {} after balancing'.format(pos_data.shape[0],
                                                                                          neg_data.shape[0]))
    balanced_data.iloc[:,category_ind] = balanced_data.iloc[:,category_ind].astype('category')
    
    return balanced_data


if __name__ == '__main__':
    input_data = pd.read_csv(sys.argv[1], header=0, sep='\t')
    label_ind = int(sys.argv[2])
    input_data = balance_data(input_data, label_ind,14)
    input_data.to_csv(sys.argv[3], sep='\t', index=False)
# END: OWN CODE
