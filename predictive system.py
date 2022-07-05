

import numpy as np
import pickle

input_data = (0.7869, 0.00615,0.0225,0.004571, 25.53, 0.243, 0.3613,0.08758)

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Gaoiz/OneDrive/Bureau/Master Data/DEPLOYMENT/trained_model.sav', 'rb'))

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is cancer benign ')
else:
  print('The person is cancer malign')