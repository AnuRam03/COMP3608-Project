import seaborn as sns
import matplotlib.pyplot as plt
    
#get the general information abou the dataset loaded
def get_dataframe_info(dataframe, title="Data Frame"):
    dbtable = dataframe
    
    print("Information of the " + title + " Data Frame\n")
    
    print("> Size of Data Frame")
    print(dbtable.shape) #show how much columns and rows in the data frame

    print("\n\n> General Information on each Column\n")
    dbtable.info() #give a general view of what the data looks like for each column

#to check for any null values within the dataset
def check_for_null_values(dataframe, title="Data Frame"):
    dbtable = dataframe

    print("Checking for Null values within the " + title + " Data Frame\n")
    
    print("> Checking Rows\n")
    print(dbtable.isna().all()) #check to see if there is any rows are empty

    print("\n\n> Checking in the Columns\n")
    print(dbtable.isna().any()) #check to see if any value in each row is missing

    print("\n\n> Summary of Missing Values Per Column\n")
    print(dbtable.isna().sum()) #return a sum of null number from each column

#tailored for fraud project - COMP3608
#plot a bar graph comparing the counts of positive and false fraud cases
def plot_class_count_bar_graph(title, xLabel, yLabel, dataset, xValue):
    full_title = "Bar Graph Showing " + title

    plt.title(full_title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    sns.countplot(x=xValue, data=dataset)
    plt.show()

    print("Summary of Graph based on Class:")
    print(dataset[xValue].value_counts())


#plot a bar graph with the count of records from each aspect
def plot_top_fraud_counts_barh(title, dataset, column_name, yLabel, xLabel, drop_values):
    counts = dataset.corr()[column_name].abs().sort_values(ascending=False).drop(drop_values)
    
    plt.figure(figsize=(10, 10))
    plt.barh(counts.index, counts.values, color="pink")
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.title(title)
    plt.gca().invert_yaxis()  # Puts the highest count on top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
   