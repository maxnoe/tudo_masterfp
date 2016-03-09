import pandas as pd


def read_data(signal_file, background_file):
    df_signal = pd.read_csv(signal_file, sep=';')
    df_signal.dropna(axis=[1, 0], how='all', inplace=True)
    print('Number of signal Features: {}. Label {}'.format(
        len(df_signal.columns), df_signal.label.iloc[0])
    )

    df_background = pd.read_csv(background_file, sep=';')
    df_background.dropna(axis=[1, 0], how='all', inplace=True)
    print('Number of background features: {}. Label: {}'.format(
        len(df_background.columns), df_background.label.iloc[0])
    )

    df = pd.concat([df_signal, df_background], axis=0, join='inner')
    return df


def drop_useless(df):
    # lets only take columns that appear in both datasets

    # match  all columns containing the name 'corsika'
    c = df.filter(regex='((C|c)orsika)(\w*[\._\-]\w*)?').columns
    df = df.drop(c, axis=1)

    # match any column containing header, MC, utc, mjd, date or ID.
    # Im sure there are better regexes for this.
    c = df.filter(regex='\w*[\.\-_]*(((u|U)(t|T)(c|C))|(MC)|((m|M)(j|J)(d|D))|(Weight)|((h|H)eader)|((d|D)ate)|(ID))+\w*[\.\-_]*\w*').columns
    df = df.drop(c, axis=1)

    # drop columns containing only a single value
    df = df.loc[:, df.var() != 0]

    # drop nans
    df = df.dropna()
    print('Combined Features: {}'.format(len(df.columns)))
    return df
