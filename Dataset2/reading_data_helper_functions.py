import pandas as pd

#function to get the file type
def get_suffix(filename, preferred_suffix=None):
    if preferred_suffix is None:
        preferred_suffix = ['.csv', '.xml', '.parquet']

    #print(preferred_suffix)

    prefix, dot, suffix = filename.rpartition('.')
    #print(parts)

    if dot == "" or "." + suffix not in preferred_suffix:
        #print('.' + suffix)
        print("File input '" + filename + "' is not accepted.")
        return None
    else:
        suffix = '.' + suffix
        return suffix

#read the relevant datafiles 
def read_dataset_file(filename, preferred_suffix=None):

    #need to determine the file format 
    #assuming they are in one of the three formats: csv, xml, parquet
    suffix = get_suffix(filename, preferred_suffix)

    if preferred_suffix is None:
        if suffix == '.csv':
            return pd.read_csv(filename)
        elif suffix == '.xml':
            return pd.read_xml(filename)
        elif suffix == '.parquet':
            return pd.read_parquet(filename)
    elif suffix in preferred_suffix:
        print(f"File reader for '{filename}' could not be found.")
        return None
    else:
        print(f"File '{filename}' is Invalid.")
        return None

    # return dbtable

#get a view of how the dataset looks
def get_head_with_pandas(dbtable):
    if dbtable is None:
        print("Dataframe is empty. Cannot process further.")
        return None
    else:
        return dbtable.head(10)