# モジュールの読み込み
# pandasをよみこみ

# train testを　dataframeにいれる
train_df = pd.read_csv(ここにtrain.csvのパスをいれる)
# 同様にtestをdataframeにいれる
X_test =

# train_df　を 説明変数(X_train)に目的変数を(y_train)にいれる。
'''
X_trainは　trainから目的変数を抜いたものを
y_trainはtrainから目的変数だけを
'''
X_train = train_df.drop(ここに目的変数をいれる, axis=1)  # axis=1は縦方向の削除
y_train = train_df[目的変数の名前]

# データをみるとGenderだけ文字列なので、int or float　に変換する
'''
replace または、 LabelEncoderを使うとよい
'''
# trainに
X_train['Gender'] = X_train['Gender'].replace()
X_train['Gender'] = X_train['Gender'].replace('Female', 0)

# testに
X_test['Gender'] = X_test['Gender'].replace('Male', 1)
X_test['Gender'] = X_test['Gender'].replace(,)
#
