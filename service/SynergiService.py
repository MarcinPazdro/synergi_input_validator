import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#import pyodbc
import sqlalchemy as sal
import pickle
from domain.domain import SynergiVerificationRequest, SynergiVerificationValue

class SynergiService():
    def __init__(self):
        self.path_encoder = "/Users/marcin/Documents/GitHub/synergi_input_validator/artifacts/synergi_input_encoder.pkl"
        self.path_model = "/Users/marcin/Documents/GitHub/synergi_input_validator/artifacts/RandomForestRegressor_For_Synergi.pkl"
        self.model = self.load_artifact(self.path_model)
        self.le = self.load_artifact(self.path_encoder)

    def load_artifact(self, path_to_artifact):
        ''' LOAD A PREDICTION MODEL FROM A PICKLE FILE '''
        with open(path_to_artifact, 'rb') as f:
            artifact = pickle.load(f)
        #    print(f"{path_to_artifact} laoded")
        return artifact

    def preprocess_input(self, request: SynergiVerificationRequest)->pd.DataFrame:
        data_dict = {
            "year": request.year, 
            "month": request.month, 
            "location":request.location, 
            "substance":request.substance 
        }
        data_df = pd.DataFrame.from_dict([data_dict])

        data_df.month = data_df.month.str.lower()
        data_df.month = self.le.fit_transform(data_df.month)
        data_df.month = data_df.month.astype('category')
        data_df.location = data_df.location.str.lower()
        data_df.location = self.le.fit_transform(data_df.location)
        data_df.location = data_df.location.astype('category')
        data_df.substance = data_df.substance.str.lower()
        data_df.substance = self.le.fit_transform(data_df.substance)
        data_df.substance = data_df.substance.astype('category')

        return data_df
    
    def predict_value(self, request: SynergiVerificationRequest)->SynergiVerificationValue:
        input_df = self.preprocess_input(request)
        # Predict
        predicted_amount = self.model.predict(input_df)[0]

        response = SynergiVerificationValue
        response.year = request.year
        response.month = request.month
        response.location = request.location
        response.substance = request.substance
        response.amount = predicted_amount
        return response

if __name__ == "__main__":
    test_request = SynergiVerificationRequest(
        year = 2021, 
        month = "01", 
        location = "Aker Solutions locations - Norway - Verdal", 
        substance = "Gasoline"
    )

    syn_serv = SynergiService()
    res = syn_serv.predict_value(request = test_request)
    print(res)