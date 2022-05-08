import pandas as pd

data = pd.DataFrame

known_views = {'Forrest Gump': 10000000,
               'The Usual Suspects': 7500000,
               'Rear Window': 6000000,
               'North by Northwest': 4000000,
               'The Secret in Their Eyes': 3000000,
               'Spotlight': 1000000}

data['view'] = data['Title'].map(known_views)
data["Year"] = 2022 - data["Year"]
data['Rating Count'] = data['Rating Count'].apply(lambda x: x.replace(",", "")).astype(int)
limited_data = data[~(data['view'].isna())]

training_data = limited_data.iloc[:, 2:]
