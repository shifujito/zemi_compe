import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# import lightgbm as lgb
import optuna.integration.lightgbm as lgb
import datetime


class ConpePipeline:
    def __init__(self) -> None:
        self.data_dir = './data'
        self.train_path = os.path.join(self.data_dir, 'train.csv')
        self.test_path = os.path.join(self.data_dir, 'test.csv')

        self.params = {
            # 二値分類問題
            'objective': 'binary',
            # AUC の最大化を目指す
            'metric': 'auc',
            # Fatal の場合出力
            'verbosity': -1,
        }
        self.columns = [
            'Age',
            'Gender',
            'T_Bil',
            'D_Bil',
            'ALP',
            'ALT_GPT',
            'AST_GOT',
            'TP',
            'Alb',
            'AG_ratio']

        self.dt_now = datetime.datetime.now()
        self.now_time = str(self.dt_now.month) + '_' + str(self.dt_now.day) + \
            '_' + str(self.dt_now.hour) + str(self.dt_now.minute)

    def pp(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        # あとできりなは使用testのtargetに2を付与
        test_df['disease'] = 2

        all_df = pd.concat([train_df, test_df])

        all_df = self._pp_age(all_df)

        return all_df

    def _pp_age(self, all_df):
        age_lr = LabelEncoder()
        all_df['Gender'] = age_lr.fit_transform(all_df['Gender'])
        return all_df

    def _model(self, all_df):
        train_df = all_df[:891]
        test_df = all_df[891:][self.columns]  # tail id is 1272

        X_train, X_val, y_train, y_val = train_test_split(
            train_df[self.columns], train_df['disease'], test_size=0.3, random_state=0)

        print(y_train)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val)

        model = lgb.train(self.params, lgb_train, valid_sets=lgb_eval,
                          verbose_eval=50,  # 50イテレーション毎に学習結果出力
                          num_boost_round=100000,  # 最大イテレーション回数指定
                          early_stopping_rounds=300
                          )

        return model, test_df, X_val, y_val

    def float_to_int(self, y_pred):
        ap_pred = []
        for i in y_pred:
            if i > 0.5:
                ans = 1
            else:
                ans = 0
            ap_pred.append(ans)
        return ap_pred

    def _predict(self, model, test_df):
        y_pred = model.predict(test_df).tolist()
        ap_pred = self.float_to_int(y_pred)
        return ap_pred

    def submit(self, y_pred):
        start_ids = 891
        ids = [i for i in range(start_ids, start_ids + len(y_pred))]
        submit_df = pd.DataFrame({'id': ids, 'target': y_pred})
        submit_df.to_csv(
            'submit' +
            self.now_time +
            '.csv',
            index=None,
            header=False)
        return None

    def confirm_vali(self, model, X, y):
        pred = self._predict(model, X)
        print('validのauc', roc_auc_score(y, pred))
        return None

    def main(self):
        all_df = self.pp()
        model, test_df, X_val, y_val = self._model(all_df)
        y_pred = self._predict(model, test_df)
        self.submit(y_pred)

        self.confirm_vali(model, X_val, y_val)


if __name__ == '__main__':
    ins = ConpePipeline()
    ins.main()
