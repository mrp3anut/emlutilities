import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def resizer(hdf5_data_path, eq_catalog_csv_path, size, output_name='resized'):

    eq_catalog = pd.read_csv(eq_catalog_csv_path)
    hdf5_data = h5py.File(hdf5_data_path,'r')

    sampled_hdf5, sampled_eq_catalog = resize(hdf5_data, eq_catalog, size, output_name)

    sampled_hdf5.close()
    sampled_eq_catalog.to_csv('{}_{}.csv'.format(output_name, str(size)))

def resize(hdf5_data, eq_catalog, size, output_name):

    ''' Takes a hdf5 file and a pandas dataframe which contains info about the hdf5 file, like a catalog.
    Outputs resized and shuffled verisons of the original hdf5 object and pandas dataframe.
    Size can take integer values between 0 and 100.'''

    data = hdf5_data['data']
    
    # Shuffle csv to randomize trace_name order and save the resultig csv
    sampled_eq_catalog = eq_catalog.sample(frac=size/100).reset_index(drop=True)
    ev_list = sampled_eq_catalog['trace_name'].to_list()
    sampled_hdf5 = h5py.File('{}_{}.hdf5'.format(output_name, size),'w')
    small_data = sampled_hdf5.create_group("data")
    
    # Copy the files from the original data to the new hdf5 object using the downsized event list.
    for c, evi in enumerate(ev_list):
        data.copy(evi,small_data)

    return sampled_hdf5, sampled_eq_catalog

def result_metrics(test_results_dict, output_name='test_result_metrics', output_path=None):
    
    result_metrics = metrics(test_results_dict)
    if output_path != None:
        result_metrics.to_csv("{}.csv".format(output_path), index=False) 
    else:
        result_metrics.to_csv("{}.csv".format(output_name), index=False) 


PROBABILITIES = ["detection_probability", 
                 "P_probability", 
                 "S_probability"]

TEST_COLUMNS = ["model_name",
                "det_recall", 
                "det_precision", 
                "d_tp", 
                "d_fp", 
                "d_tn", 
                "d_fn", 
                "p_recall", 
                "p_precision", 
                "p_mae", 
                "p_rmse", 
                "p_tp", 
                "p_fp", 
                "p_tn", 
                "p_fn", 
                "s_recall", 
                "s_precision", 
                "s_mae", 
                "s_rmse", 
                "s_tp", 
                "s_fp", 
                "s_tn", 
                "s_fn", 
                "#events", 
                "#noise"]

def metrics(test_results_dict):

    ''' test_results_dict should be in this format:
        keys are model names,
        values are pandas dataframes which contain test result csv files from EQTransformer's tester output. '''

    result_metrics = pd.DataFrame()


    for model in test_results_dict:
        
        results = []
        results.append(model)
        
        for probability in PROBABILITIES:
            TP=0
            FP=0
            TN=0
            FN=0
            event = 0
            not_event = 0
            
            earthquakes = test_results_dict[model][test_results_dict[model]["trace_category"] == "earthquake_local"]
            event = len(earthquakes)
            nan_pred_event = earthquakes[earthquakes["{}".format(probability)].isnull()]
            FN = len(nan_pred_event)
            TP = event-FN
            
            noise = test_results_dict[model][test_results_dict[model]["trace_category"] == "noise"]
            not_event = len(noise)
            nan_pred_noise = noise[noise["{}".format(probability)].isnull()]
            TN = len(nan_pred_noise)
            FP = not_event-TN
                        
            recall = TP/(TP+FN)
            precision = TP/(TP+FP)
            
            eq_dropna_p = earthquakes[~earthquakes["P_pick"].isnull()]
            eq_dropna = eq_dropna_p[~eq_dropna_p["S_pick"].isnull()]
            results.append(recall)
            results.append(precision)
            
            if probability == "P_probability":
                mae = mean_absolute_error(eq_dropna["P_pick"],eq_dropna["p_arrival_sample"])
                rmse = mean_squared_error(eq_dropna["P_pick"],eq_dropna["p_arrival_sample"])**(1/2)
                results.append(mae)
                results.append(rmse)
                
            if probability == "S_probability":   
                mae = mean_absolute_error(eq_dropna["S_pick"],eq_dropna["s_arrival_sample"])
                rmse = mean_squared_error(eq_dropna["S_pick"],eq_dropna["s_arrival_sample"])**(1/2)
                results.append(mae)
                results.append(rmse)
                
            results.append(TP)
            results.append(FP)
            results.append(TN)
            results.append(FN)
            
        results.append(event)
        results.append(not_event)
        result_metrics = result_metrics.append(pd.DataFrame(results).T)
        
    result_metrics.columns = TEST_COLUMNS
    result_metrics = result_metrics.reset_index()
    result_metrics = result_metrics.drop("index", axis=1)

    return result_metrics

def comparison(test_result_metrics_csv_path, model_to_compare_to='EQT', output_name="comparison_catalog", output_path=None):
    
    test_result_metrics = pd.read_csv(test_result_metrics_csv_path, index_col=0)
    better_parameters = compare(test_result_metrics, model_to_compare_to)
    
    if output_path != None:
        better_parameters.to_csv("{}.csv".format(output_path)) 
    else:
        better_parameters.to_csv("{}.csv".format(output_name)) 


GOOD_COLUMNS = ["det_recall", 
                "det_precision",
                "d_tp",
                "d_tn",
                "p_recall",
                "p_precision",
                "p_tp",
                "p_tn",
                "s_recall",
                "s_precision",
                "s_tp",
                "s_tn"]

BAD_COLUMNS = ["d_fp",
               "d_fn",
               "p_fp",
               "p_fn",
               "s_fp",
               "s_fn",
               "p_mae",
               "p_rmse",
               "s_mae",
               "s_rmse"]

def compare(test_result_metrics, model_to_compare_to='EQT'):

    '''Takes a Pandas DataFrame object in the format of what metrics() returns'''
    better_parameters = pd.DataFrame(index=test_result_metrics.index, columns=test_result_metrics.columns)
                             
    for model in test_result_metrics.index:
        better_list=[]
        for parameter in GOOD_COLUMNS:
            if test_result_metrics.loc[model][parameter] > test_result_metrics.loc[model_to_compare_to][parameter]:
                better_parameters.loc[model, parameter] = "Better"
            elif test_result_metrics.loc[model][parameter] == test_result_metrics.loc[model_to_compare_to][parameter]:
                better_parameters.loc[model, parameter] = "Equal"
            else:
                better_parameters.loc[model, parameter] = "Worse"

        for parameter in BAD_COLUMNS:
            if test_result_metrics.loc[model][parameter] < test_result_metrics.loc[model_to_compare_to][parameter]:
                better_parameters.loc[model, parameter] = "Better"
            elif test_result_metrics.loc[model][parameter] == test_result_metrics.loc[model_to_compare_to][parameter]:
                better_parameters.loc[model, parameter] = "Equal"
            else:
                better_parameters.loc[model, parameter] = "Worse"

    return better_parameters




