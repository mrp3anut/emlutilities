import pandas as pd
import numpy as np
import h5py
import os
from emlutilities.core import resize, result_metrics, compare

def test_resizer():

    eq_catalog = pd.read_csv("ModelsAndSampleData/100samples.csv")
    hdf5_data = h5py.File("ModelsAndSampleData/100samples.hdf5", 'r')

    output_name = "resized"
    size=50

    sampled_hdf5, sampled_eq_catalog = resize(hdf5_data, eq_catalog, size, output_name)

    os.remove("{}_{}.hdf5".format(output_name, size))

    assert len(sampled_eq_catalog) == 50

def test_results_metrics():

    eqt = pd.read_csv("ModelsAndSampleData/X_test_results_EQT.csv")
    test_dict = {'EQT':eqt}
    test_columns = np.array(["model_name","det_recall","det_precision","d_tp","d_fp","d_tn","d_fn","p_recall","p_precision","p_mae","p_rmse","p_tp","p_fp","p_tn","p_fn","s_recall","s_precision","s_mae","s_rmse","s_tp","s_fp","s_tn","s_fn","#events","#noise"])
    result_metrics(test_dict)
    
    csv = pd.read_csv("test_result_metrics.csv")
    
    comparison = csv.columns == test_columns
    equal_arrays = comparison.all()

    assert equal_arrays

    os.remove('test_result_metrics.csv')

    
def test_compare():
    dataframe = pd.read_csv("ModelsAndSampleData/test_results.csv", index_col=0)
    a = compare(dataframe)
    assert a.loc["test"]['det_recall'] == 'Better'



