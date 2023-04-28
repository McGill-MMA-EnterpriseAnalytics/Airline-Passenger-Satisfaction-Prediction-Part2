!pip install ctgan



!pip install table_evaluator

from google.colab import files
uploaded = files.upload()

import io
import pandas as pd
data = pd.read_csv(io.BytesIO(uploaded['train.csv']))
# Dataset is now stored in a Pandas Dataframe
data = data.drop('Unnamed: 0', axis=1)
data.dropna(inplace=True)

columns_list = data.columns.tolist()
strings_to_remove = ["id","Age","Flight Distance"]

new_list = [item for item in columns_list if item not in strings_to_remove]
new_list

categorical_features = new_list


from ctgan import CTGAN
import datetime as dt
ctgan = CTGAN(verbose=True)
ctgan.fit(data, categorical_features, epochs = 5)

samples = ctgan.sample(1000)

samples


import pandas as pd
import pickle

pickle.dump(ctgan, open('Datagenerator.pkl', 'wb'))

# testing to generate data
# import datetime as dt
# import pickle
# import pandas as pd
# pickled_model = pickle.load(open('Datagenerator.pkl', 'rb'))
# newsamples = pickled_model.sample(1000)
# newsamples["TimeStamp"]=dt.datetime.now()

# newsamples = pickled_model.sample(1000)

# newsamples


# newsamples["TimeStamp"]=dt.datetime.now()

# newsamples

