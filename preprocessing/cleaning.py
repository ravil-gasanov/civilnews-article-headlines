import pandas as pd

def clean(raw):
    # drop a garbage column 
    raw.drop(raw.columns[-1], axis = 1, inplace = True)

    # drop rows that have missing values in either headline or views columns
    mask = raw[['headline', 'views']].dropna().index
    raw = raw.loc[mask, :]

    raw['date-time'] = pd.to_datetime(raw['date-time'], format = "%d/%m/%Y - %H:%M")
    raw.reset_index(inplace = True, drop = True)

    raw.sort_values(by = ['date-time'], ascending = True, inplace = True)
    assert raw['date-time'].is_monotonic, "not sorted"

    raw = raw.loc[pd.to_numeric(raw['views'], errors = 'coerce').dropna().index, :]
    raw['views'] = raw['views'].astype(float)

    return raw.reset_index(drop = True)