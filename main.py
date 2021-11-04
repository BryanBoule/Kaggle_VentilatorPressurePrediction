import warnings
from sklearn.preprocessing import RobustScaler
from helpers.methods import create_features
from kfold import k_fold
from cfg import Config
import pandas as pd
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # Load train
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    train_df = create_features(train_df)
    test_df = create_features(test_df)

    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    # robustscaler is great to take into account outliers

    RS = RobustScaler()
    # not including u_out as it is a boolean
    columns_to_scale = [elm for elm in train_df.columns if
                        elm not in ['id', 'breath_id', 'time_step', 'pressure', 'u_out']]
    train_df[columns_to_scale] = RS.fit_transform(train_df[columns_to_scale])
    test_df['pressure'] = 0
    test_df[columns_to_scale] = RS.transform(test_df[columns_to_scale])
    test_df = test_df.drop('pressure', axis=1)

    pred_oof, pred_test = k_fold(
        Config,
        train_df,
        test_df,
    )

    sub = pd.read_csv('./data/sample_submission.csv')
    sub['pressure'] = pred_test
    sub.to_csv('./data/submission.csv', index=False)
