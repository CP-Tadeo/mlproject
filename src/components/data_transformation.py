import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:
            ##temp
            # numerical_columns = ['writing_score','reading_score']
            # categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            #

            num_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                           ("scaler",StandardScaler())
                                           
                                           ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder",OneHotEncoder()),
                #("scaler",StandardScaler(with_mean=False)) #optional
            ])


            logging.info("Encodings completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def get_data_target_transformer_object(self, target_column, df):
        try:
            ##temp
            # numerical_columns = ['writing_score','reading_score']
            # categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            #

            num_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                           ("scaler",StandardScaler())
                                           
                                           ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder",OneHotEncoder()),
                #("scaler",StandardScaler(with_mean=False)) #optional
            ])


            logging.info("Encodings completed")

            if pd.api.types.is_object_dtype(df[target_column]):
                return LabelEncoder()
            else:
                num_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                           ("scaler",StandardScaler())
                                           
                                           ])
                return num_pipeline
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        ##
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(list(train_df.columns))

            col_dict = {i: col for i, col in enumerate(train_df.select_dtypes(exclude="object").columns)}

            while True:
                for key, value in col_dict.items():
                    print("{}: {}".format((key+1), value))

                # col_dict = {i: col for i, col in enumerate(test_df.columns)}

                # for key, value in col_dict.items():
                #     print("{}: {}".format((key+1), value))
                col_to_remove = int(input("Input corresponding column number to remove:\n"))-1

                if 0<= col_to_remove < len(col_dict):
                    break
                else:
                    print("Invalid selection")
                

            target_column_name = col_dict[col_to_remove]

            logging.info("reading train and test data complete; processing with obtaining preprocessing object")

            #target_column_name="math_score"
            #numerical_columns = ['writing_score','reading_score']

            input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df = train_df[target_column_name]

            #if pd.api.types.is_object_dtype(target_feature_train_df):


            input_feature_test_df = test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df = test_df[target_column_name]

            num_features = input_feature_train_df.select_dtypes(exclude="object").columns
            cat_features = input_feature_train_df.select_dtypes(include="object").columns
            print("num features: {}".format(num_features))
            print("cat features: {}".format(cat_features))

            logging.info("applying preprocesing")

            preprocessing_obj=self.get_data_transformer_object(numerical_columns=num_features, categorical_columns=cat_features)
            

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            #
            
            #
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_Arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saved preprocessing object")

            save_object(file_path=self.data_transformation_config.preprocessor_ob_file_path,
                        obj=preprocessing_obj)


            return (train_arr, test_Arr, self.data_transformation_config.preprocessor_ob_file_path)
        except Exception as e:
            raise CustomException(e,sys)