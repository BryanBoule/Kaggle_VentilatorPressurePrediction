import numpy as np
from torch.utils.data import Dataset
import torch


class VPPDataLoader(Dataset):
    def __init__(self, dataframe):
        if "pressure" not in dataframe.columns:
            dataframe['pressure'] = 0

        # aggregate data and store features as list
        self.df_grouped = dataframe.groupby('breath_id').agg(list).reset_index()
        self._preprocess()

    def _preprocess(self):
        self.pressures = np.array(self.df_grouped['pressure'].values.tolist())
        rs = np.array(self.df_grouped['R'].values.tolist())
        cs = np.array(self.df_grouped['C'].values.tolist())
        u_ins = np.array(self.df_grouped['u_in'].values.tolist())
        self.u_outs = np.array(self.df_grouped['u_out'].values.tolist())

        # u_in_lag_fwrd1 = np.array(self.df_grouped['u_in_lag_fwrd1'].values.tolist())
        # u_in_lag_back1 = np.array(self.df_grouped['u_in_lag_back1'].values.tolist())
        # u_in_lag_fwrd2 = np.array(self.df_grouped['u_in_lag_fwrd2'].values.tolist())
        # u_in_lag_back2 = np.array(self.df_grouped['u_in_lag_back2'].values.tolist())
        # u_in_lag_fwrd3 = np.array(self.df_grouped['u_in_lag_fwrd3'].values.tolist())
        # u_in_lag_back3 = np.array(self.df_grouped['u_in_lag_back3'].values.tolist())
        # u_in_lag_fwrd4 = np.array(self.df_grouped['u_in_lag_fwrd4'].values.tolist())
        # u_in_lag_back4 = np.array(self.df_grouped['u_in_lag_back4'].values.tolist())
        # time_diff = np.array(self.df_grouped['time_diff'].values.tolist())
        # time_diff_2 = np.array(self.df_grouped['time_diff_2'].values.tolist())
        # time_diff_3 = np.array(self.df_grouped['time_diff_3'].values.tolist())
        # area = np.array(self.df_grouped['area'].values.tolist())
        # area_cumsum = np.array(self.df_grouped['area_cumsum'].values.tolist())
        # auc_u_in = np.array(self.df_grouped['auc_u_in'].values.tolist())
        # auc_u_in_cumsum = np.array(self.df_grouped['auc_u_in_cumsum'].values.tolist())
        # u_in_max = np.array(self.df_grouped['u_in_max'].values.tolist())
        # u_in_min = np.array(self.df_grouped['u_in_min'].values.tolist())
        # u_in_mean = np.array(self.df_grouped['u_in_mean'].values.tolist())
        # u_in_median = np.array(self.df_grouped['u_in_median'].values.tolist())
        # u_in_range = np.array(self.df_grouped['u_in_range'].values.tolist())

        # [:, None] increases array dimension from 1 to 2,
        # becomes a [[v1, v2, v3]] numpy array
        self.inputs = np.concatenate([
            rs[:, None],
            cs[:, None],
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None],
            # u_in_lag_fwrd1[:, None],
            # u_in_lag_back1[:, None],
            # u_in_lag_fwrd2[:, None],
            # u_in_lag_back2[:, None],
            # u_in_lag_fwrd3[:, None],
            # u_in_lag_back3[:, None],
            # u_in_lag_fwrd4[:, None],
            # u_in_lag_back4[:, None],
            # time_diff[:, None],
            # time_diff_2[:, None],
            # time_diff_3[:, None],
            # area[:, None],
            # area_cumsum[:, None],
            # auc_u_in[:, None],
            # auc_u_in_cumsum[:, None],
            # u_in_max[:, None],
            # u_in_min[:, None],
            # u_in_mean[:, None],
            # u_in_median[:, None],
            # u_in_range[:, None],
        ], axis=1).transpose(0, 2, 1)

    def __len__(self):
        return self.df_grouped.shape[0]

    def __getitem__(self, index):
        target = 'pressure'
        return {
            "input": torch.tensor(self.inputs[index], dtype=torch.float),
            "u_out": torch.tensor(self.df_grouped.u_out[index], dtype=torch.int8),
            "p": torch.tensor(self.df_grouped.loc[index, target], dtype=torch.float),
        }
