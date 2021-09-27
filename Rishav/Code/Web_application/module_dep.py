import pandas as pd
import datetime

class DataStream:
    
    df_base_data = None
    
    def initialize_data(self):
        """
        Function to read the source (base) data and type convert date column
        :return: None
        """
        filename_base_data = 'base_data_resampled_tomek.csv'
        self.df_base_data = pd.read_csv(filename_base_data)
        self.df_base_data.date = self.df_base_data.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
        return None

    def get_data(self, filter_date):
        """
        :param filter_date: the date for which the data is requested, data type: datetime.date object
        :return: sliced dataframe
        """
        df_filtered_data = self.df_base_data.loc[self.df_base_data.date==filter_date]
        return df_filtered_data
    
    
###############################################################################