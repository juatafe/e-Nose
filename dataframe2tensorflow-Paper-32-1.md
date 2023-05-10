```python
#imports aquí 
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/talens/ml4bda/')
import caixaBigot as cb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import subprocess
import time
import scipy
import seaborn as sns
import h5py
from pandas import DataFrame
from pandas import Series
from pandas import HDFStore,DataFrame
from pandas import Series
import re
import listArxiu
import arreglaCurva
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from scipy import signal
import pickle as pl #http://fredborg-braedstrup.dk/blog/2014/10/10/saving-mpl-figures-using-pickle/
import os
import representar
import glob
import shutil
import COVs
import correctorMoosy32
import contarTipus
from engineering_notation import EngNumber#pip3 install engineering_notation
import arff
from TGS import TGS
import param2dfX2 as param
import pandas as pd
from dfX2h5 import *
import os
import sys
import tensorflow as tf
import torch #afegir a la documentacio
```

    2022-11-27 08:53:27.369865: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-11-27 08:53:27.538040: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2022-11-27 08:53:27.538079: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2022-11-27 08:53:27.573006: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2022-11-27 08:53:28.315272: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
    2022-11-27 08:53:28.315387: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
    2022-11-27 08:53:28.315396: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.



```python
'''Anem a carregar el dataframe de train en memòria'''
directory = './salida/'
hdf_dirname =directory+'CaP-PBH-40-train.h5'
df1=pd.read_hdf(hdf_dirname, '/df1')
#df1.head
df1.sample(frac=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sensor</th>
      <th>el75</th>
      <th>std</th>
      <th>moda</th>
      <th>media</th>
      <th>mediana</th>
      <th>iqr</th>
      <th>cv</th>
      <th>V40</th>
      <th>V60</th>
      <th>...</th>
      <th>asimetria</th>
      <th>Met</th>
      <th>IsoB</th>
      <th>Prop</th>
      <th>Hidro</th>
      <th>Etan</th>
      <th>CO</th>
      <th>Air</th>
      <th>class</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>S10-TGS2600</td>
      <td>0.150819</td>
      <td>0.069291</td>
      <td>0.189258</td>
      <td>0.085089</td>
      <td>0.072641</td>
      <td>0.132080</td>
      <td>81.433326</td>
      <td>0.069711</td>
      <td>0.146082</td>
      <td>...</td>
      <td>0.262400</td>
      <td>1.106776e-06</td>
      <td>6.173166e-03</td>
      <td>0.000000</td>
      <td>0.010546</td>
      <td>0.005950</td>
      <td>0.005326</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>S21-TGS2611-e00</td>
      <td>0.006369</td>
      <td>0.003019</td>
      <td>0.008873</td>
      <td>0.003729</td>
      <td>0.003168</td>
      <td>0.005664</td>
      <td>80.955206</td>
      <td>0.003895</td>
      <td>0.006844</td>
      <td>...</td>
      <td>0.316835</td>
      <td>5.067252e+01</td>
      <td>1.312942e+15</td>
      <td>0.000000</td>
      <td>97.789777</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>S11-TGS2620</td>
      <td>0.035076</td>
      <td>0.016256</td>
      <td>0.044457</td>
      <td>0.020156</td>
      <td>0.018590</td>
      <td>0.029532</td>
      <td>80.655059</td>
      <td>0.015143</td>
      <td>0.033614</td>
      <td>...</td>
      <td>0.169460</td>
      <td>3.320687e+02</td>
      <td>1.423500e+01</td>
      <td>0.000000</td>
      <td>7.492807</td>
      <td>14.152548</td>
      <td>35.202036</td>
      <td>0.000000e+00</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>S14-TGS2610-c00</td>
      <td>0.005801</td>
      <td>0.003046</td>
      <td>0.008158</td>
      <td>0.003349</td>
      <td>0.002508</td>
      <td>0.005168</td>
      <td>90.956926</td>
      <td>0.001960</td>
      <td>0.004957</td>
      <td>...</td>
      <td>0.580695</td>
      <td>7.636273e+01</td>
      <td>7.102536e+01</td>
      <td>71.025359</td>
      <td>152.807578</td>
      <td>306.011841</td>
      <td>0.000000</td>
      <td>2.890314e+127</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>S17-TGS2611-c00</td>
      <td>0.032393</td>
      <td>0.041098</td>
      <td>0.132407</td>
      <td>0.028871</td>
      <td>0.014237</td>
      <td>0.029911</td>
      <td>142.348639</td>
      <td>0.026914</td>
      <td>0.048561</td>
      <td>...</td>
      <td>1.395665</td>
      <td>9.452639e+01</td>
      <td>1.000000e+00</td>
      <td>0.000000</td>
      <td>266.164640</td>
      <td>461.253299</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>S31-TGS2620</td>
      <td>0.009769</td>
      <td>0.005156</td>
      <td>0.011653</td>
      <td>0.005748</td>
      <td>0.005583</td>
      <td>0.008296</td>
      <td>89.689105</td>
      <td>0.002513</td>
      <td>0.007355</td>
      <td>...</td>
      <td>0.183405</td>
      <td>2.691814e+02</td>
      <td>1.211062e+01</td>
      <td>0.000000</td>
      <td>6.340062</td>
      <td>12.347678</td>
      <td>30.034116</td>
      <td>0.000000e+00</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>S10-TGS2600</td>
      <td>0.212338</td>
      <td>0.094646</td>
      <td>0.261079</td>
      <td>0.119615</td>
      <td>0.107266</td>
      <td>0.185240</td>
      <td>79.125961</td>
      <td>0.133565</td>
      <td>0.218978</td>
      <td>...</td>
      <td>0.157426</td>
      <td>1.482349e-06</td>
      <td>6.725313e-03</td>
      <td>0.000000</td>
      <td>0.011352</td>
      <td>0.006462</td>
      <td>0.005983</td>
      <td>0.000000e+00</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>S26-TGS2600</td>
      <td>0.196432</td>
      <td>0.075508</td>
      <td>0.222090</td>
      <td>0.132037</td>
      <td>0.142099</td>
      <td>0.101846</td>
      <td>57.187479</td>
      <td>0.108138</td>
      <td>0.182462</td>
      <td>...</td>
      <td>-0.502102</td>
      <td>1.385344e-06</td>
      <td>6.593174e-03</td>
      <td>0.000000</td>
      <td>0.011160</td>
      <td>0.006340</td>
      <td>0.005824</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S8-TGS2600</td>
      <td>0.127742</td>
      <td>0.050175</td>
      <td>0.140362</td>
      <td>0.087812</td>
      <td>0.100800</td>
      <td>0.061319</td>
      <td>57.139135</td>
      <td>0.077488</td>
      <td>0.119270</td>
      <td>...</td>
      <td>-0.761494</td>
      <td>9.647440e-07</td>
      <td>5.929516e-03</td>
      <td>0.000000</td>
      <td>0.010187</td>
      <td>0.005724</td>
      <td>0.005043</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>S12-TGS2611-c00</td>
      <td>0.014301</td>
      <td>0.008567</td>
      <td>0.023304</td>
      <td>0.007816</td>
      <td>0.004258</td>
      <td>0.014007</td>
      <td>109.596561</td>
      <td>0.004361</td>
      <td>0.013254</td>
      <td>...</td>
      <td>0.775298</td>
      <td>8.171921e+01</td>
      <td>1.000000e+00</td>
      <td>0.000000</td>
      <td>223.830378</td>
      <td>390.435151</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>12800 rows × 34 columns</p>
</div>




```python
'''Canvie el tipus a float32 per si de cas'''
df1[df1.select_dtypes(np.float64).columns] = df1.select_dtypes(np.float64).astype(np.float32)
df1[df1.select_dtypes(np.int64).columns] = df1.select_dtypes(np.int64).astype(np.float32)
df1.dtypes
```




    Sensor        object
    el75         float32
    std          float32
    moda         float32
    media        float32
    mediana      float32
    iqr          float32
    cv           float32
    V40          float32
    V60          float32
    Vmax         float32
    V100         float32
    V120         float32
    difBA        float32
    difBC        float32
    difBD        float32
    difBE        float32
    slopeAB      float32
    slopeBC      float32
    slopeAD      float32
    slopeDE      float32
    slopeEC      float32
    slopeBE      float32
    slopeDB      float32
    asimetria    float32
    Met          float32
    IsoB         float32
    Prop         float32
    Hidro        float32
    Etan         float32
    CO           float32
    Air          float32
    class        float32
    target       float32
    dtype: object




```python
#df1.isnull().sum().sum()
df1.fillna(0.0)
df1.replace([np.inf, -np.inf], np.nan, inplace=True)
#df1.replace(np.nan, 0)


```


```python
df1.isnull().sum().sum()
```




    829




```python
'''Per tal de canviar de text a números '''
sensor_mapping={label:idx for idx, label in
               enumerate (np.unique(df1['Sensor']))} 
sensor_mapping

df1['Sensor']=df1['Sensor'].map(sensor_mapping) #canvia el valor
```


```python
'''Per tal de revertir el número a text'''
#inv_sensor_mapping = {v: k for k, v in sensor_mapping.items()}
# df1['Sensor']= df1['Sensor'].map(inv_class_mapping) 
```




    'Per tal de revertir el número a text'




```python
#target = df1.iloc[0:,32:34].values
#print('Class labels:', np.unique(target))
#target.shape
```


```python
target = df1.pop('target')
#target.head()
#print('Class labels:', np.unique(target))
```


```python
numeric_feature_names = ['Sensor',
                         'el75',
                         'std',
                         'moda',
                         'media',
                         'mediana',
                         'iqr',
                         'cv',
                         'V40',
                         'V60',
                         'Vmax',
                         'V100',
                         'V120',
                         'difBA',
                         'difBC',
                         'difBD',
                         'difBE',
                         'slopeAB',
                         'slopeBC',
                         'slopeAD',
                         'slopeDE',
                         'slopeEC',
                         'slopeBE',
                         'slopeDB',
                         'asimetria',
                         'Met',
                         'IsoB',
                         'Prop',
                         'Hidro',
                         'Etan',
                         'CO',
                         'Air',
                        ]
numeric_features = df1[numeric_feature_names]
numeric_features.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sensor</th>
      <th>el75</th>
      <th>std</th>
      <th>moda</th>
      <th>media</th>
      <th>mediana</th>
      <th>iqr</th>
      <th>cv</th>
      <th>V40</th>
      <th>V60</th>
      <th>...</th>
      <th>slopeBE</th>
      <th>slopeDB</th>
      <th>asimetria</th>
      <th>Met</th>
      <th>IsoB</th>
      <th>Prop</th>
      <th>Hidro</th>
      <th>Etan</th>
      <th>CO</th>
      <th>Air</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.003686</td>
      <td>0.001450</td>
      <td>0.004288</td>
      <td>0.002404</td>
      <td>0.002489</td>
      <td>0.002410</td>
      <td>60.334152</td>
      <td>0.001459</td>
      <td>0.003232</td>
      <td>...</td>
      <td>-0.000043</td>
      <td>0.000053</td>
      <td>-0.159998</td>
      <td>6.542502e+01</td>
      <td>1.000000e+00</td>
      <td>0.0</td>
      <td>171.792999</td>
      <td>302.673584</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>0.015796</td>
      <td>0.008602</td>
      <td>0.022377</td>
      <td>0.009319</td>
      <td>0.006354</td>
      <td>0.013158</td>
      <td>92.301987</td>
      <td>0.004017</td>
      <td>0.012348</td>
      <td>...</td>
      <td>-0.000545</td>
      <td>0.000547</td>
      <td>0.719136</td>
      <td>2.055548e-07</td>
      <td>3.768247e-03</td>
      <td>0.0</td>
      <td>0.006898</td>
      <td>0.003698</td>
      <td>0.002726</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>0.049925</td>
      <td>0.021870</td>
      <td>0.062636</td>
      <td>0.029701</td>
      <td>0.027040</td>
      <td>0.037673</td>
      <td>73.633072</td>
      <td>0.027291</td>
      <td>0.050004</td>
      <td>...</td>
      <td>-0.000976</td>
      <td>0.000689</td>
      <td>0.113840</td>
      <td>3.533174e+02</td>
      <td>1.493114e+01</td>
      <td>0.0</td>
      <td>7.871860</td>
      <td>14.734585</td>
      <td>36.892464</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>0.014490</td>
      <td>0.006116</td>
      <td>0.017908</td>
      <td>0.008826</td>
      <td>0.008261</td>
      <td>0.010453</td>
      <td>69.294838</td>
      <td>0.008389</td>
      <td>0.014617</td>
      <td>...</td>
      <td>-0.000259</td>
      <td>0.000180</td>
      <td>0.036221</td>
      <td>8.190455e+01</td>
      <td>1.000000e+00</td>
      <td>0.0</td>
      <td>224.434525</td>
      <td>391.449127</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>0.004684</td>
      <td>0.002020</td>
      <td>0.005858</td>
      <td>0.002793</td>
      <td>0.002583</td>
      <td>0.003706</td>
      <td>72.325844</td>
      <td>0.002845</td>
      <td>0.004770</td>
      <td>...</td>
      <td>-0.000089</td>
      <td>0.000060</td>
      <td>0.121335</td>
      <td>4.802394e+01</td>
      <td>1.447402e+14</td>
      <td>0.0</td>
      <td>91.416496</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
'''Per a Normalitzar les dades per files'''
#numeric_features=numeric_features.div(numeric_features.sum(axis=1), axis=0)
#numeric_features.head()
#numeric_features.max()
```




    'Per a Normalitzar les dades per files'




```python

```


```python
'''Convertir el Dataframe en tensor'''
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(numeric_features)
tensor_train=tf.convert_to_tensor(numeric_features)
tensor_target=tf.convert_to_tensor(target)
def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)
tensor_train = replacenan(tensor_train)
tensor_target=replacenan(tensor_target)
#un warning amb la cpu si no tens gpu https://stackoverflow.com/questions/59499764/tensorflow-not-tensorflow-gpu-failed-call-to-cuinit-unknown-error-303
#tf.convert_to_tensor(target)

#numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))
#for row in numeric_dict_ds.take(3):
#  print(row)
print(tensor_train)
```

    tf.Tensor(
    [[0.00000000e+00 3.68634262e-03 1.45034480e-03 ... 3.02673584e+02
      0.00000000e+00 0.00000000e+00]
     [1.10000000e+01 1.57958083e-02 8.60165339e-03 ... 3.69828404e-03
      2.72589596e-03 0.00000000e+00]
     [2.20000000e+01 4.99251634e-02 2.18698382e-02 ... 1.47345848e+01
      3.68924637e+01 0.00000000e+00]
     ...
     [2.30000000e+01 9.52446461e-02 4.25203294e-02 ... 1.59189215e+01
      4.03656807e+01 0.00000000e+00]
     [2.40000000e+01 8.37306008e-02 3.74861844e-02 ... 1.56689968e+01
      3.96291084e+01 0.00000000e+00]
     [2.50000000e+01 7.69510493e-02 3.46375629e-02 ... 5.10707404e-03
      4.29488625e-03 0.00000000e+00]], shape=(12800, 32), dtype=float64)


    2022-11-27 08:53:47.623409: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2022-11-27 08:53:47.623445: W tensorflow/stream_executor/cuda/cuda_driver.cc:263] failed call to cuInit: UNKNOWN ERROR (303)
    2022-11-27 08:53:47.623471: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (talens): /proc/driver/nvidia/version does not exist
    2022-11-27 08:53:47.623799: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
'''def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)'''

'''numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

for row in numeric_dataset.take(3):
  print(row)
'''

```




    'numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))\n\nfor row in numeric_dataset.take(3):\n  print(row)\n'




```python
#model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)

numeric_features.shape
```




    (12800, 32)




```python
'''Normalitzar les dades per a que tots els inputs sumen 1 creem el normalitzador'''
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(tensor_train) 
#norm_x = normalizer(numeric_features.replace(np.nan,0))
#print(tf.reduce_mean(norm_x), tf.math.reduce_std(norm_x))
#print(norm_x)
```


```python
target.shape
```




    (12800,)




```python
def get_basic_model():
  model = tf.keras.Sequential([
      #tf.keras.layers.Flatten(input_shape=(32,)),
      tf.keras.layers.Input(shape=(32,)),#input_dim=32, units=50, activation='tanh'),
      normalizer,
      tf.keras.layers.Dense(input_dim=32, units=64, activation='relu'),
      tf.keras.layers.Dense(input_dim=64, units=32, activation='relu'),
      tf.keras.layers.Dense(input_dim=32, units=16, activation='relu'),
      tf.keras.layers.Dense(input_dim=16, units=2, activation='softmax'),#'sigmoid')
  ])

  #model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
  #              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#,#BinaryCrossentropy(from_logits=True),
  #              metrics=['accuracy']
  #             )
  
  '''model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics=['accuracy'])'''
  '''model.compile(optimizer='sgd',
               loss='binary_crossentropy',
               metrics=['accuracy'])'''
  
  sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-7, momentum=.9)

  model.compile(optimizer=sgd_optimizer,
              loss='categorical_crossentropy',
               metrics=['accuracy'])
  

  return model
```


```python
#model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)

```


```python
#numeric_features.head()
```


```python
#numeric_features.replace(np.nan,0)
#print(target.values)

#target=np.array(target,ndmin=2)
#target=pd.DataFrame(np.array(df1.pop['target'],ndmin=2), columns=['target'])
#target
tensor_train
```




    <tf.Tensor: shape=(12800, 32), dtype=float64, numpy=
    array([[0.00000000e+00, 3.68634262e-03, 1.45034480e-03, ...,
            3.02673584e+02, 0.00000000e+00, 0.00000000e+00],
           [1.10000000e+01, 1.57958083e-02, 8.60165339e-03, ...,
            3.69828404e-03, 2.72589596e-03, 0.00000000e+00],
           [2.20000000e+01, 4.99251634e-02, 2.18698382e-02, ...,
            1.47345848e+01, 3.68924637e+01, 0.00000000e+00],
           ...,
           [2.30000000e+01, 9.52446461e-02, 4.25203294e-02, ...,
            1.59189215e+01, 4.03656807e+01, 0.00000000e+00],
           [2.40000000e+01, 8.37306008e-02, 3.74861844e-02, ...,
            1.56689968e+01, 3.96291084e+01, 0.00000000e+00],
           [2.50000000e+01, 7.69510493e-02, 3.46375629e-02, ...,
            5.10707404e-03, 4.29488625e-03, 0.00000000e+00]])>




```python
SHUFFLE_BUFFER = 0
BATCH_SIZE = 32
target_onehot=tf.keras.utils.to_categorical(tensor_target)
model = get_basic_model()
class_weight = {0: 1.,
                1: 32.}
historial=model.fit(tensor_train, target_onehot, epochs=1280, verbose=True, batch_size=BATCH_SIZE,class_weight=class_weight)
print('model entrenat!')
```

    Epoch 1/1280
    400/400 [==============================] - 1s 2ms/step - loss: 2.2776 - accuracy: 0.5001
    Epoch 2/1280
    400/400 [==============================] - 1s 2ms/step - loss: 2.0463 - accuracy: 0.5034
    Epoch 3/1280
    400/400 [==============================] - 1s 2ms/step - loss: 2.0047 - accuracy: 0.5074
    Epoch 4/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.9665 - accuracy: 0.5095
    Epoch 5/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.9489 - accuracy: 0.5091
    Epoch 6/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.9311 - accuracy: 0.5098
    Epoch 7/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.9096 - accuracy: 0.5109
    Epoch 8/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.8956 - accuracy: 0.5110
    Epoch 9/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.8835 - accuracy: 0.5123
    Epoch 10/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.8836 - accuracy: 0.5119
    Epoch 11/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.8589 - accuracy: 0.5130
    Epoch 12/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.8519 - accuracy: 0.5143
    Epoch 13/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.8441 - accuracy: 0.5141
    Epoch 14/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.8232 - accuracy: 0.5165
    Epoch 15/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.8113 - accuracy: 0.5173
    Epoch 16/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.8033 - accuracy: 0.5177
    Epoch 17/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7928 - accuracy: 0.5196
    Epoch 18/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7773 - accuracy: 0.5198
    Epoch 19/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7614 - accuracy: 0.5261
    Epoch 20/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7601 - accuracy: 0.5280
    Epoch 21/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7497 - accuracy: 0.5304
    Epoch 22/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7393 - accuracy: 0.5355
    Epoch 23/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7160 - accuracy: 0.5438
    Epoch 24/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7097 - accuracy: 0.5473
    Epoch 25/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7113 - accuracy: 0.5521
    Epoch 26/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7000 - accuracy: 0.5514
    Epoch 27/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6892 - accuracy: 0.5580
    Epoch 28/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6957 - accuracy: 0.5600
    Epoch 29/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6768 - accuracy: 0.5677
    Epoch 30/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6598 - accuracy: 0.5723
    Epoch 31/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6893 - accuracy: 0.5676
    Epoch 32/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6717 - accuracy: 0.5705
    Epoch 33/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6440 - accuracy: 0.5759
    Epoch 34/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6419 - accuracy: 0.5759
    Epoch 35/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6246 - accuracy: 0.5839
    Epoch 36/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6159 - accuracy: 0.5866
    Epoch 37/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6214 - accuracy: 0.5865
    Epoch 38/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6247 - accuracy: 0.5869
    Epoch 39/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6169 - accuracy: 0.5888
    Epoch 40/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6024 - accuracy: 0.5927
    Epoch 41/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6033 - accuracy: 0.5957
    Epoch 42/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5906 - accuracy: 0.5991
    Epoch 43/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5873 - accuracy: 0.5989
    Epoch 44/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6163 - accuracy: 0.5967
    Epoch 45/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5967 - accuracy: 0.5987
    Epoch 46/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5770 - accuracy: 0.6047
    Epoch 47/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5639 - accuracy: 0.6099
    Epoch 48/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5699 - accuracy: 0.6096
    Epoch 49/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5816 - accuracy: 0.6079
    Epoch 50/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5622 - accuracy: 0.6088
    Epoch 51/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5834 - accuracy: 0.6110
    Epoch 52/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5716 - accuracy: 0.6099
    Epoch 53/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5426 - accuracy: 0.6160
    Epoch 54/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5608 - accuracy: 0.6149
    Epoch 55/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5701 - accuracy: 0.6120
    Epoch 56/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5106 - accuracy: 0.6227
    Epoch 57/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5382 - accuracy: 0.6209
    Epoch 58/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5165 - accuracy: 0.6218
    Epoch 59/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5239 - accuracy: 0.6253
    Epoch 60/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5170 - accuracy: 0.6294
    Epoch 61/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5501 - accuracy: 0.6241
    Epoch 62/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5175 - accuracy: 0.6266
    Epoch 63/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5102 - accuracy: 0.6312
    Epoch 64/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4761 - accuracy: 0.6384
    Epoch 65/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4860 - accuracy: 0.6385
    Epoch 66/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5115 - accuracy: 0.6333
    Epoch 67/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5035 - accuracy: 0.6349
    Epoch 68/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5458 - accuracy: 0.6249
    Epoch 69/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4951 - accuracy: 0.6316
    Epoch 70/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4658 - accuracy: 0.6413
    Epoch 71/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4715 - accuracy: 0.6413
    Epoch 72/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4794 - accuracy: 0.6395
    Epoch 73/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4444 - accuracy: 0.6486
    Epoch 74/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4660 - accuracy: 0.6443
    Epoch 75/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4965 - accuracy: 0.6381
    Epoch 76/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5409 - accuracy: 0.6326
    Epoch 77/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4734 - accuracy: 0.6424
    Epoch 78/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5185 - accuracy: 0.6365
    Epoch 79/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5252 - accuracy: 0.6317
    Epoch 80/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4727 - accuracy: 0.6455
    Epoch 81/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4349 - accuracy: 0.6497
    Epoch 82/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5054 - accuracy: 0.6378
    Epoch 83/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4718 - accuracy: 0.6439
    Epoch 84/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4505 - accuracy: 0.6468
    Epoch 85/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4194 - accuracy: 0.6564
    Epoch 86/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4887 - accuracy: 0.6478
    Epoch 87/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4540 - accuracy: 0.6514
    Epoch 88/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4488 - accuracy: 0.6460
    Epoch 89/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4103 - accuracy: 0.6590
    Epoch 90/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4295 - accuracy: 0.6558
    Epoch 91/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4206 - accuracy: 0.6555
    Epoch 92/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4198 - accuracy: 0.6609
    Epoch 93/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4342 - accuracy: 0.6559
    Epoch 94/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4370 - accuracy: 0.6615
    Epoch 95/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4530 - accuracy: 0.6498
    Epoch 96/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3827 - accuracy: 0.6672
    Epoch 97/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4127 - accuracy: 0.6622
    Epoch 98/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4811 - accuracy: 0.6446
    Epoch 99/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4104 - accuracy: 0.6581
    Epoch 100/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4269 - accuracy: 0.6609
    Epoch 101/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4156 - accuracy: 0.6654
    Epoch 102/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3957 - accuracy: 0.6635
    Epoch 103/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3842 - accuracy: 0.6680
    Epoch 104/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3611 - accuracy: 0.6745
    Epoch 105/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3755 - accuracy: 0.6741
    Epoch 106/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3320 - accuracy: 0.6822
    Epoch 107/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3860 - accuracy: 0.6765
    Epoch 108/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3804 - accuracy: 0.6732
    Epoch 109/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4215 - accuracy: 0.6626
    Epoch 110/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4167 - accuracy: 0.6638
    Epoch 111/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3686 - accuracy: 0.6711
    Epoch 112/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3652 - accuracy: 0.6739
    Epoch 113/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3766 - accuracy: 0.6720
    Epoch 114/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3797 - accuracy: 0.6709
    Epoch 115/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3957 - accuracy: 0.6723
    Epoch 116/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3456 - accuracy: 0.6779
    Epoch 117/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3420 - accuracy: 0.6798
    Epoch 118/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3398 - accuracy: 0.6841
    Epoch 119/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3516 - accuracy: 0.6820
    Epoch 120/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3795 - accuracy: 0.6781
    Epoch 121/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3510 - accuracy: 0.6813
    Epoch 122/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3290 - accuracy: 0.6854
    Epoch 123/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3349 - accuracy: 0.6866
    Epoch 124/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3804 - accuracy: 0.6774
    Epoch 125/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4317 - accuracy: 0.6670
    Epoch 126/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3353 - accuracy: 0.6803
    Epoch 127/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3213 - accuracy: 0.6865
    Epoch 128/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3467 - accuracy: 0.6805
    Epoch 129/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3224 - accuracy: 0.6820
    Epoch 130/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3314 - accuracy: 0.6885
    Epoch 131/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3785 - accuracy: 0.6798
    Epoch 132/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3373 - accuracy: 0.6840
    Epoch 133/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2887 - accuracy: 0.6913
    Epoch 134/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3132 - accuracy: 0.6909
    Epoch 135/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3326 - accuracy: 0.6890
    Epoch 136/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2938 - accuracy: 0.6953
    Epoch 137/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2741 - accuracy: 0.6983
    Epoch 138/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3389 - accuracy: 0.6886
    Epoch 139/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3448 - accuracy: 0.6804
    Epoch 140/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2984 - accuracy: 0.6962
    Epoch 141/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3212 - accuracy: 0.6907
    Epoch 142/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3066 - accuracy: 0.6980
    Epoch 143/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3104 - accuracy: 0.6920
    Epoch 144/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3170 - accuracy: 0.6933
    Epoch 145/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3475 - accuracy: 0.6863
    Epoch 146/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3127 - accuracy: 0.6916
    Epoch 147/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2923 - accuracy: 0.6948
    Epoch 148/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3124 - accuracy: 0.6917
    Epoch 149/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2812 - accuracy: 0.6949
    Epoch 150/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2604 - accuracy: 0.7034
    Epoch 151/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2905 - accuracy: 0.6957
    Epoch 152/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3113 - accuracy: 0.6931
    Epoch 153/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4764 - accuracy: 0.6702
    Epoch 154/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4053 - accuracy: 0.6728
    Epoch 155/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3621 - accuracy: 0.6812
    Epoch 156/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3524 - accuracy: 0.6803
    Epoch 157/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2752 - accuracy: 0.7006
    Epoch 158/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2437 - accuracy: 0.7084
    Epoch 159/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2578 - accuracy: 0.7027
    Epoch 160/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2438 - accuracy: 0.7059
    Epoch 161/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2155 - accuracy: 0.7129
    Epoch 162/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3986 - accuracy: 0.6809
    Epoch 163/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4007 - accuracy: 0.6645
    Epoch 164/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3261 - accuracy: 0.6884
    Epoch 165/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2688 - accuracy: 0.7016
    Epoch 166/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2720 - accuracy: 0.7049
    Epoch 167/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2814 - accuracy: 0.7029
    Epoch 168/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2192 - accuracy: 0.7154
    Epoch 169/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2366 - accuracy: 0.7118
    Epoch 170/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2164 - accuracy: 0.7134
    Epoch 171/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2384 - accuracy: 0.7112
    Epoch 172/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3013 - accuracy: 0.7017
    Epoch 173/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2619 - accuracy: 0.7012
    Epoch 174/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2312 - accuracy: 0.7166
    Epoch 175/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2231 - accuracy: 0.7187
    Epoch 176/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2451 - accuracy: 0.7100
    Epoch 177/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2497 - accuracy: 0.7134
    Epoch 178/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2246 - accuracy: 0.7176
    Epoch 179/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3007 - accuracy: 0.7054
    Epoch 180/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3151 - accuracy: 0.6959
    Epoch 181/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3631 - accuracy: 0.6770
    Epoch 182/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4371 - accuracy: 0.6823
    Epoch 183/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2590 - accuracy: 0.7026
    Epoch 184/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2261 - accuracy: 0.7135
    Epoch 185/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2007 - accuracy: 0.7173
    Epoch 186/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1903 - accuracy: 0.7205
    Epoch 187/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2370 - accuracy: 0.7166
    Epoch 188/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2715 - accuracy: 0.7066
    Epoch 189/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2075 - accuracy: 0.7161
    Epoch 190/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2051 - accuracy: 0.7194
    Epoch 191/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2184 - accuracy: 0.7155
    Epoch 192/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1669 - accuracy: 0.7276
    Epoch 193/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1917 - accuracy: 0.7273
    Epoch 194/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6960 - accuracy: 0.6530
    Epoch 195/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3861 - accuracy: 0.6727
    Epoch 196/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3742 - accuracy: 0.6823
    Epoch 197/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3459 - accuracy: 0.6884
    Epoch 198/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2443 - accuracy: 0.7105
    Epoch 199/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2342 - accuracy: 0.7140
    Epoch 200/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3751 - accuracy: 0.6849
    Epoch 201/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3431 - accuracy: 0.6758
    Epoch 202/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.5765 - accuracy: 0.6454
    Epoch 203/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3568 - accuracy: 0.6695
    Epoch 204/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2848 - accuracy: 0.6905
    Epoch 205/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2609 - accuracy: 0.6991
    Epoch 206/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3267 - accuracy: 0.6886
    Epoch 207/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2894 - accuracy: 0.6971
    Epoch 208/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2620 - accuracy: 0.7044
    Epoch 209/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2556 - accuracy: 0.7102
    Epoch 210/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2930 - accuracy: 0.7026
    Epoch 211/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2221 - accuracy: 0.7093
    Epoch 212/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3155 - accuracy: 0.6934
    Epoch 213/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2578 - accuracy: 0.7027
    Epoch 214/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2425 - accuracy: 0.7084
    Epoch 215/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1930 - accuracy: 0.7235
    Epoch 216/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1484 - accuracy: 0.7348
    Epoch 217/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1641 - accuracy: 0.7310
    Epoch 218/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1882 - accuracy: 0.7252
    Epoch 219/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1762 - accuracy: 0.7302
    Epoch 220/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2936 - accuracy: 0.7100
    Epoch 221/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2869 - accuracy: 0.7040
    Epoch 222/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2116 - accuracy: 0.7224
    Epoch 223/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2104 - accuracy: 0.7230
    Epoch 224/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2235 - accuracy: 0.7191
    Epoch 225/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1919 - accuracy: 0.7230
    Epoch 226/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1203 - accuracy: 0.7419
    Epoch 227/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1903 - accuracy: 0.7295
    Epoch 228/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1850 - accuracy: 0.7219
    Epoch 229/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1301 - accuracy: 0.7406
    Epoch 230/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1656 - accuracy: 0.7341
    Epoch 231/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2851 - accuracy: 0.7124
    Epoch 232/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1643 - accuracy: 0.7287
    Epoch 233/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3755 - accuracy: 0.7045
    Epoch 234/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4608 - accuracy: 0.6523
    Epoch 235/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2902 - accuracy: 0.6931
    Epoch 236/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2062 - accuracy: 0.7145
    Epoch 237/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2032 - accuracy: 0.7237
    Epoch 238/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2042 - accuracy: 0.7229
    Epoch 239/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1679 - accuracy: 0.7284
    Epoch 240/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1638 - accuracy: 0.7332
    Epoch 241/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2706 - accuracy: 0.7084
    Epoch 242/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2390 - accuracy: 0.7155
    Epoch 243/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1978 - accuracy: 0.7239
    Epoch 244/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1085 - accuracy: 0.7416
    Epoch 245/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1602 - accuracy: 0.7337
    Epoch 246/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1973 - accuracy: 0.7261
    Epoch 247/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1444 - accuracy: 0.7385
    Epoch 248/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1875 - accuracy: 0.7289
    Epoch 249/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1466 - accuracy: 0.7391
    Epoch 250/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2383 - accuracy: 0.7252
    Epoch 251/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2234 - accuracy: 0.7188
    Epoch 252/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2132 - accuracy: 0.7191
    Epoch 253/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1235 - accuracy: 0.7362
    Epoch 254/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1350 - accuracy: 0.7393
    Epoch 255/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1289 - accuracy: 0.7420
    Epoch 256/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1244 - accuracy: 0.7424
    Epoch 257/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2397 - accuracy: 0.7182
    Epoch 258/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1707 - accuracy: 0.7299
    Epoch 259/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2456 - accuracy: 0.7195
    Epoch 260/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1779 - accuracy: 0.7316
    Epoch 261/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1711 - accuracy: 0.7316
    Epoch 262/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1328 - accuracy: 0.7405
    Epoch 263/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1365 - accuracy: 0.7366
    Epoch 264/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1460 - accuracy: 0.7392
    Epoch 265/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1019 - accuracy: 0.7454
    Epoch 266/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2215 - accuracy: 0.7239
    Epoch 267/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3505 - accuracy: 0.6953
    Epoch 268/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1604 - accuracy: 0.7330
    Epoch 269/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1271 - accuracy: 0.7391
    Epoch 270/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1434 - accuracy: 0.7387
    Epoch 271/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1307 - accuracy: 0.7423
    Epoch 272/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1565 - accuracy: 0.7371
    Epoch 273/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1360 - accuracy: 0.7389
    Epoch 274/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0631 - accuracy: 0.7533
    Epoch 275/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1133 - accuracy: 0.7441
    Epoch 276/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0728 - accuracy: 0.7546
    Epoch 277/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0500 - accuracy: 0.7582
    Epoch 278/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0832 - accuracy: 0.7543
    Epoch 279/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1175 - accuracy: 0.7452
    Epoch 280/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3568 - accuracy: 0.7136
    Epoch 281/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2401 - accuracy: 0.7147
    Epoch 282/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1479 - accuracy: 0.7358
    Epoch 283/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1704 - accuracy: 0.7295
    Epoch 284/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1169 - accuracy: 0.7405
    Epoch 285/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1285 - accuracy: 0.7401
    Epoch 286/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0617 - accuracy: 0.7563
    Epoch 287/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0747 - accuracy: 0.7527
    Epoch 288/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1482 - accuracy: 0.7435
    Epoch 289/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4091 - accuracy: 0.6809
    Epoch 290/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1314 - accuracy: 0.7327
    Epoch 291/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1838 - accuracy: 0.7270
    Epoch 292/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1488 - accuracy: 0.7298
    Epoch 293/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1208 - accuracy: 0.7439
    Epoch 294/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1793 - accuracy: 0.7318
    Epoch 295/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0905 - accuracy: 0.7505
    Epoch 296/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0829 - accuracy: 0.7498
    Epoch 297/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0691 - accuracy: 0.7565
    Epoch 298/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1110 - accuracy: 0.7500
    Epoch 299/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1079 - accuracy: 0.7505
    Epoch 300/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1626 - accuracy: 0.7344
    Epoch 301/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1473 - accuracy: 0.7445
    Epoch 302/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2875 - accuracy: 0.7149
    Epoch 303/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1564 - accuracy: 0.7343
    Epoch 304/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2603 - accuracy: 0.7144
    Epoch 305/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1126 - accuracy: 0.7434
    Epoch 306/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2396 - accuracy: 0.7276
    Epoch 307/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2538 - accuracy: 0.7091
    Epoch 308/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1800 - accuracy: 0.7280
    Epoch 309/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1408 - accuracy: 0.7326
    Epoch 310/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1231 - accuracy: 0.7379
    Epoch 311/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2021 - accuracy: 0.7320
    Epoch 312/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3652 - accuracy: 0.7069
    Epoch 313/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2278 - accuracy: 0.7163
    Epoch 314/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1080 - accuracy: 0.7448
    Epoch 315/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0767 - accuracy: 0.7510
    Epoch 316/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1467 - accuracy: 0.7390
    Epoch 317/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1538 - accuracy: 0.7296
    Epoch 318/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1337 - accuracy: 0.7412
    Epoch 319/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0378 - accuracy: 0.7588
    Epoch 320/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2613 - accuracy: 0.7176
    Epoch 321/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2130 - accuracy: 0.7255
    Epoch 322/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2033 - accuracy: 0.7208
    Epoch 323/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1686 - accuracy: 0.7355
    Epoch 324/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1071 - accuracy: 0.7439
    Epoch 325/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0847 - accuracy: 0.7548
    Epoch 326/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1026 - accuracy: 0.7452
    Epoch 327/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0941 - accuracy: 0.7472
    Epoch 328/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0938 - accuracy: 0.7455
    Epoch 329/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1599 - accuracy: 0.7394
    Epoch 330/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0999 - accuracy: 0.7453
    Epoch 331/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1483 - accuracy: 0.7380
    Epoch 332/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1536 - accuracy: 0.7333
    Epoch 333/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2228 - accuracy: 0.7171
    Epoch 334/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1538 - accuracy: 0.7334
    Epoch 335/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0610 - accuracy: 0.7512
    Epoch 336/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1618 - accuracy: 0.7397
    Epoch 337/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0834 - accuracy: 0.7523
    Epoch 338/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0836 - accuracy: 0.7518
    Epoch 339/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1898 - accuracy: 0.7325
    Epoch 340/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1204 - accuracy: 0.7444
    Epoch 341/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1787 - accuracy: 0.7355
    Epoch 342/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0740 - accuracy: 0.7507
    Epoch 343/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2117 - accuracy: 0.7273
    Epoch 344/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2663 - accuracy: 0.7178
    Epoch 345/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2078 - accuracy: 0.7191
    Epoch 346/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1198 - accuracy: 0.7405
    Epoch 347/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0855 - accuracy: 0.7538
    Epoch 348/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0746 - accuracy: 0.7532
    Epoch 349/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0301 - accuracy: 0.7623
    Epoch 350/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0340 - accuracy: 0.7652
    Epoch 351/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0577 - accuracy: 0.7574
    Epoch 352/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0249 - accuracy: 0.7670
    Epoch 353/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0018 - accuracy: 0.7704
    Epoch 354/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0603 - accuracy: 0.7613
    Epoch 355/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0656 - accuracy: 0.7563
    Epoch 356/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0487 - accuracy: 0.7595
    Epoch 357/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1911 - accuracy: 0.7400
    Epoch 358/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1871 - accuracy: 0.7290
    Epoch 359/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0568 - accuracy: 0.7562
    Epoch 360/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0731 - accuracy: 0.7578
    Epoch 361/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0006 - accuracy: 0.7688
    Epoch 362/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0045 - accuracy: 0.7729
    Epoch 363/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9879 - accuracy: 0.7729
    Epoch 364/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9768 - accuracy: 0.7763
    Epoch 365/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1213 - accuracy: 0.7473
    Epoch 366/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1818 - accuracy: 0.7401
    Epoch 367/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1776 - accuracy: 0.7309
    Epoch 368/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0307 - accuracy: 0.7617
    Epoch 369/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1485 - accuracy: 0.7400
    Epoch 370/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0437 - accuracy: 0.7572
    Epoch 371/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0413 - accuracy: 0.7617
    Epoch 372/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1211 - accuracy: 0.7447
    Epoch 373/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0275 - accuracy: 0.7622
    Epoch 374/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0661 - accuracy: 0.7563
    Epoch 375/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1514 - accuracy: 0.7389
    Epoch 376/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0745 - accuracy: 0.7526
    Epoch 377/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0617 - accuracy: 0.7609
    Epoch 378/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0260 - accuracy: 0.7632
    Epoch 379/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9841 - accuracy: 0.7739
    Epoch 380/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0590 - accuracy: 0.7591
    Epoch 381/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0281 - accuracy: 0.7655
    Epoch 382/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0605 - accuracy: 0.7607
    Epoch 383/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1269 - accuracy: 0.7447
    Epoch 384/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1521 - accuracy: 0.7340
    Epoch 385/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0783 - accuracy: 0.7524
    Epoch 386/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3123 - accuracy: 0.7227
    Epoch 387/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1542 - accuracy: 0.7298
    Epoch 388/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1522 - accuracy: 0.7379
    Epoch 389/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0763 - accuracy: 0.7505
    Epoch 390/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0466 - accuracy: 0.7582
    Epoch 391/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0492 - accuracy: 0.7574
    Epoch 392/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0880 - accuracy: 0.7533
    Epoch 393/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1305 - accuracy: 0.7423
    Epoch 394/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0180 - accuracy: 0.7625
    Epoch 395/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0414 - accuracy: 0.7613
    Epoch 396/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0152 - accuracy: 0.7669
    Epoch 397/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1055 - accuracy: 0.7511
    Epoch 398/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3754 - accuracy: 0.7059
    Epoch 399/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2059 - accuracy: 0.7126
    Epoch 400/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1968 - accuracy: 0.7230
    Epoch 401/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1418 - accuracy: 0.7309
    Epoch 402/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1724 - accuracy: 0.7236
    Epoch 403/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0709 - accuracy: 0.7478
    Epoch 404/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0251 - accuracy: 0.7634
    Epoch 405/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0584 - accuracy: 0.7538
    Epoch 406/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1012 - accuracy: 0.7459
    Epoch 407/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0688 - accuracy: 0.7502
    Epoch 408/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0280 - accuracy: 0.7633
    Epoch 409/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1325 - accuracy: 0.7450
    Epoch 410/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1488 - accuracy: 0.7392
    Epoch 411/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0387 - accuracy: 0.7603
    Epoch 412/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9943 - accuracy: 0.7695
    Epoch 413/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0121 - accuracy: 0.7684
    Epoch 414/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1117 - accuracy: 0.7495
    Epoch 415/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0589 - accuracy: 0.7599
    Epoch 416/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1952 - accuracy: 0.7355
    Epoch 417/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1089 - accuracy: 0.7412
    Epoch 418/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0256 - accuracy: 0.7626
    Epoch 419/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0582 - accuracy: 0.7592
    Epoch 420/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0772 - accuracy: 0.7520
    Epoch 421/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0284 - accuracy: 0.7595
    Epoch 422/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9948 - accuracy: 0.7723
    Epoch 423/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9988 - accuracy: 0.7710
    Epoch 424/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0468 - accuracy: 0.7617
    Epoch 425/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0740 - accuracy: 0.7556
    Epoch 426/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0256 - accuracy: 0.7680
    Epoch 427/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0590 - accuracy: 0.7585
    Epoch 428/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0098 - accuracy: 0.7657
    Epoch 429/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0016 - accuracy: 0.7705
    Epoch 430/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3266 - accuracy: 0.7123
    Epoch 431/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1175 - accuracy: 0.7390
    Epoch 432/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0072 - accuracy: 0.7673
    Epoch 433/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9737 - accuracy: 0.7770
    Epoch 434/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0359 - accuracy: 0.7631
    Epoch 435/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0551 - accuracy: 0.7580
    Epoch 436/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9940 - accuracy: 0.7710
    Epoch 437/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0067 - accuracy: 0.7713
    Epoch 438/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1585 - accuracy: 0.7438
    Epoch 439/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1986 - accuracy: 0.7247
    Epoch 440/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2034 - accuracy: 0.7284
    Epoch 441/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0532 - accuracy: 0.7548
    Epoch 442/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1297 - accuracy: 0.7521
    Epoch 443/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1859 - accuracy: 0.7270
    Epoch 444/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9973 - accuracy: 0.7684
    Epoch 445/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0912 - accuracy: 0.7591
    Epoch 446/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0243 - accuracy: 0.7655
    Epoch 447/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0224 - accuracy: 0.7679
    Epoch 448/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0476 - accuracy: 0.7656
    Epoch 449/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0790 - accuracy: 0.7554
    Epoch 450/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0841 - accuracy: 0.7548
    Epoch 451/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2208 - accuracy: 0.7211
    Epoch 452/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0672 - accuracy: 0.7510
    Epoch 453/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0141 - accuracy: 0.7680
    Epoch 454/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0297 - accuracy: 0.7663
    Epoch 455/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0026 - accuracy: 0.7678
    Epoch 456/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0263 - accuracy: 0.7691
    Epoch 457/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0256 - accuracy: 0.7587
    Epoch 458/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9106 - accuracy: 0.7874
    Epoch 459/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0532 - accuracy: 0.7646
    Epoch 460/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0479 - accuracy: 0.7580
    Epoch 461/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1498 - accuracy: 0.7416
    Epoch 462/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0584 - accuracy: 0.7583
    Epoch 463/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1203 - accuracy: 0.7448
    Epoch 464/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0450 - accuracy: 0.7585
    Epoch 465/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1888 - accuracy: 0.7277
    Epoch 466/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0974 - accuracy: 0.7447
    Epoch 467/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9740 - accuracy: 0.7727
    Epoch 468/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0382 - accuracy: 0.7660
    Epoch 469/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0341 - accuracy: 0.7663
    Epoch 470/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9882 - accuracy: 0.7728
    Epoch 471/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0166 - accuracy: 0.7707
    Epoch 472/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0415 - accuracy: 0.7630
    Epoch 473/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9472 - accuracy: 0.7802
    Epoch 474/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9338 - accuracy: 0.7867
    Epoch 475/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0318 - accuracy: 0.7692
    Epoch 476/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0230 - accuracy: 0.7680
    Epoch 477/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0305 - accuracy: 0.7678
    Epoch 478/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9489 - accuracy: 0.7813
    Epoch 479/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9153 - accuracy: 0.7895
    Epoch 480/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9880 - accuracy: 0.7814
    Epoch 481/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1093 - accuracy: 0.7623
    Epoch 482/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2112 - accuracy: 0.7315
    Epoch 483/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0831 - accuracy: 0.7511
    Epoch 484/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9887 - accuracy: 0.7725
    Epoch 485/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1576 - accuracy: 0.7462
    Epoch 486/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1795 - accuracy: 0.7350
    Epoch 487/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0593 - accuracy: 0.7542
    Epoch 488/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0351 - accuracy: 0.7633
    Epoch 489/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0517 - accuracy: 0.7583
    Epoch 490/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9990 - accuracy: 0.7650
    Epoch 491/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9423 - accuracy: 0.7812
    Epoch 492/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9094 - accuracy: 0.7929
    Epoch 493/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9463 - accuracy: 0.7826
    Epoch 494/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1069 - accuracy: 0.7574
    Epoch 495/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0684 - accuracy: 0.7614
    Epoch 496/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0070 - accuracy: 0.7673
    Epoch 497/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1042 - accuracy: 0.7507
    Epoch 498/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0332 - accuracy: 0.7601
    Epoch 499/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9449 - accuracy: 0.7786
    Epoch 500/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9217 - accuracy: 0.7891
    Epoch 501/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9175 - accuracy: 0.7887
    Epoch 502/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0160 - accuracy: 0.7724
    Epoch 503/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9853 - accuracy: 0.7766
    Epoch 504/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2102 - accuracy: 0.7412
    Epoch 505/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0766 - accuracy: 0.7509
    Epoch 506/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9826 - accuracy: 0.7708
    Epoch 507/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0587 - accuracy: 0.7621
    Epoch 508/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0661 - accuracy: 0.7644
    Epoch 509/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0481 - accuracy: 0.7620
    Epoch 510/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1938 - accuracy: 0.7434
    Epoch 511/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0810 - accuracy: 0.7537
    Epoch 512/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0547 - accuracy: 0.7608
    Epoch 513/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0144 - accuracy: 0.7663
    Epoch 514/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9500 - accuracy: 0.7805
    Epoch 515/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9356 - accuracy: 0.7817
    Epoch 516/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9128 - accuracy: 0.7877
    Epoch 517/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1515 - accuracy: 0.7538
    Epoch 518/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1013 - accuracy: 0.7511
    Epoch 519/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0333 - accuracy: 0.7632
    Epoch 520/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0188 - accuracy: 0.7647
    Epoch 521/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9302 - accuracy: 0.7843
    Epoch 522/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9108 - accuracy: 0.7884
    Epoch 523/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0046 - accuracy: 0.7695
    Epoch 524/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9032 - accuracy: 0.7906
    Epoch 525/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8753 - accuracy: 0.8009
    Epoch 526/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0813 - accuracy: 0.7717
    Epoch 527/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1312 - accuracy: 0.7448
    Epoch 528/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0603 - accuracy: 0.7611
    Epoch 529/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0486 - accuracy: 0.7581
    Epoch 530/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9796 - accuracy: 0.7778
    Epoch 531/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9535 - accuracy: 0.7830
    Epoch 532/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9580 - accuracy: 0.7797
    Epoch 533/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1424 - accuracy: 0.7545
    Epoch 534/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1263 - accuracy: 0.7430
    Epoch 535/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0679 - accuracy: 0.7517
    Epoch 536/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9670 - accuracy: 0.7743
    Epoch 537/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9107 - accuracy: 0.7876
    Epoch 538/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9658 - accuracy: 0.7815
    Epoch 539/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9657 - accuracy: 0.7788
    Epoch 540/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0322 - accuracy: 0.7675
    Epoch 541/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9282 - accuracy: 0.7855
    Epoch 542/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8765 - accuracy: 0.7979
    Epoch 543/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9234 - accuracy: 0.7916
    Epoch 544/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9423 - accuracy: 0.7850
    Epoch 545/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0430 - accuracy: 0.7762
    Epoch 546/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2973 - accuracy: 0.7166
    Epoch 547/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9801 - accuracy: 0.7698
    Epoch 548/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9654 - accuracy: 0.7775
    Epoch 549/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9514 - accuracy: 0.7812
    Epoch 550/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9641 - accuracy: 0.7794
    Epoch 551/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1698 - accuracy: 0.7498
    Epoch 552/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1284 - accuracy: 0.7397
    Epoch 553/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0154 - accuracy: 0.7617
    Epoch 554/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0896 - accuracy: 0.7539
    Epoch 555/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9562 - accuracy: 0.7814
    Epoch 556/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9832 - accuracy: 0.7750
    Epoch 557/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8939 - accuracy: 0.7921
    Epoch 558/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9333 - accuracy: 0.7887
    Epoch 559/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9238 - accuracy: 0.7910
    Epoch 560/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8886 - accuracy: 0.7957
    Epoch 561/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2059 - accuracy: 0.7440
    Epoch 562/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9747 - accuracy: 0.7755
    Epoch 563/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9257 - accuracy: 0.7891
    Epoch 564/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9455 - accuracy: 0.7847
    Epoch 565/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9536 - accuracy: 0.7841
    Epoch 566/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9252 - accuracy: 0.7877
    Epoch 567/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9371 - accuracy: 0.7880
    Epoch 568/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9484 - accuracy: 0.7814
    Epoch 569/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0117 - accuracy: 0.7736
    Epoch 570/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0595 - accuracy: 0.7627
    Epoch 571/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0519 - accuracy: 0.7620
    Epoch 572/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0975 - accuracy: 0.7532
    Epoch 573/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9362 - accuracy: 0.7820
    Epoch 574/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8933 - accuracy: 0.7938
    Epoch 575/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8722 - accuracy: 0.7974
    Epoch 576/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0172 - accuracy: 0.7749
    Epoch 577/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0853 - accuracy: 0.7578
    Epoch 578/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0105 - accuracy: 0.7695
    Epoch 579/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9801 - accuracy: 0.7747
    Epoch 580/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9744 - accuracy: 0.7791
    Epoch 581/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9247 - accuracy: 0.7860
    Epoch 582/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8938 - accuracy: 0.7964
    Epoch 583/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8930 - accuracy: 0.7956
    Epoch 584/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0409 - accuracy: 0.7852
    Epoch 585/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1638 - accuracy: 0.7485
    Epoch 586/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0799 - accuracy: 0.7542
    Epoch 587/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0015 - accuracy: 0.7717
    Epoch 588/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9748 - accuracy: 0.7791
    Epoch 589/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9430 - accuracy: 0.7820
    Epoch 590/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0285 - accuracy: 0.7698
    Epoch 591/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9615 - accuracy: 0.7821
    Epoch 592/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9944 - accuracy: 0.7714
    Epoch 593/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9287 - accuracy: 0.7878
    Epoch 594/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1458 - accuracy: 0.7613
    Epoch 595/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3183 - accuracy: 0.7209
    Epoch 596/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0239 - accuracy: 0.7618
    Epoch 597/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0276 - accuracy: 0.7652
    Epoch 598/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0442 - accuracy: 0.7613
    Epoch 599/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9422 - accuracy: 0.7802
    Epoch 600/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9074 - accuracy: 0.7890
    Epoch 601/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8881 - accuracy: 0.7962
    Epoch 602/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9646 - accuracy: 0.7845
    Epoch 603/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9396 - accuracy: 0.7835
    Epoch 604/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9360 - accuracy: 0.7895
    Epoch 605/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0202 - accuracy: 0.7694
    Epoch 606/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9677 - accuracy: 0.7827
    Epoch 607/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0184 - accuracy: 0.7693
    Epoch 608/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9177 - accuracy: 0.7907
    Epoch 609/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0191 - accuracy: 0.7721
    Epoch 610/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8724 - accuracy: 0.7970
    Epoch 611/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8868 - accuracy: 0.7961
    Epoch 612/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9554 - accuracy: 0.7900
    Epoch 613/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3003 - accuracy: 0.7151
    Epoch 614/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0837 - accuracy: 0.7499
    Epoch 615/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9890 - accuracy: 0.7748
    Epoch 616/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9054 - accuracy: 0.7912
    Epoch 617/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9329 - accuracy: 0.7869
    Epoch 618/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0161 - accuracy: 0.7775
    Epoch 619/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0425 - accuracy: 0.7688
    Epoch 620/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0008 - accuracy: 0.7741
    Epoch 621/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0376 - accuracy: 0.7640
    Epoch 622/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9542 - accuracy: 0.7745
    Epoch 623/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9264 - accuracy: 0.7870
    Epoch 624/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8621 - accuracy: 0.8003
    Epoch 625/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8496 - accuracy: 0.8050
    Epoch 626/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0373 - accuracy: 0.7705
    Epoch 627/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9377 - accuracy: 0.7842
    Epoch 628/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8984 - accuracy: 0.7911
    Epoch 629/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9025 - accuracy: 0.7948
    Epoch 630/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0224 - accuracy: 0.7725
    Epoch 631/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9450 - accuracy: 0.7816
    Epoch 632/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9226 - accuracy: 0.7888
    Epoch 633/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1451 - accuracy: 0.7555
    Epoch 634/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9563 - accuracy: 0.7814
    Epoch 635/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8724 - accuracy: 0.7957
    Epoch 636/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8387 - accuracy: 0.8048
    Epoch 637/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8214 - accuracy: 0.8098
    Epoch 638/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8518 - accuracy: 0.8061
    Epoch 639/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8486 - accuracy: 0.8055
    Epoch 640/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8272 - accuracy: 0.8102
    Epoch 641/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9190 - accuracy: 0.7946
    Epoch 642/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8505 - accuracy: 0.8067
    Epoch 643/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1547 - accuracy: 0.7636
    Epoch 644/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0984 - accuracy: 0.7570
    Epoch 645/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9706 - accuracy: 0.7721
    Epoch 646/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8819 - accuracy: 0.7932
    Epoch 647/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8588 - accuracy: 0.8000
    Epoch 648/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9375 - accuracy: 0.7880
    Epoch 649/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8689 - accuracy: 0.8005
    Epoch 650/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0681 - accuracy: 0.7716
    Epoch 651/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9150 - accuracy: 0.7869
    Epoch 652/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8267 - accuracy: 0.8074
    Epoch 653/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8328 - accuracy: 0.8091
    Epoch 654/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8179 - accuracy: 0.8133
    Epoch 655/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8821 - accuracy: 0.7991
    Epoch 656/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9234 - accuracy: 0.7937
    Epoch 657/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0876 - accuracy: 0.7643
    Epoch 658/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0949 - accuracy: 0.7536
    Epoch 659/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9339 - accuracy: 0.7873
    Epoch 660/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8823 - accuracy: 0.7954
    Epoch 661/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9418 - accuracy: 0.7907
    Epoch 662/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0407 - accuracy: 0.7684
    Epoch 663/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9101 - accuracy: 0.7919
    Epoch 664/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9015 - accuracy: 0.7929
    Epoch 665/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8837 - accuracy: 0.7992
    Epoch 666/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9423 - accuracy: 0.7852
    Epoch 667/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8893 - accuracy: 0.7952
    Epoch 668/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9284 - accuracy: 0.7925
    Epoch 669/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2607 - accuracy: 0.7351
    Epoch 670/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9763 - accuracy: 0.7772
    Epoch 671/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8856 - accuracy: 0.7974
    Epoch 672/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9468 - accuracy: 0.7851
    Epoch 673/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0974 - accuracy: 0.7580
    Epoch 674/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2486 - accuracy: 0.7398
    Epoch 675/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1462 - accuracy: 0.7355
    Epoch 676/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1167 - accuracy: 0.7469
    Epoch 677/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0542 - accuracy: 0.7596
    Epoch 678/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9344 - accuracy: 0.7817
    Epoch 679/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9404 - accuracy: 0.7833
    Epoch 680/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9866 - accuracy: 0.7777
    Epoch 681/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9435 - accuracy: 0.7816
    Epoch 682/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8710 - accuracy: 0.7973
    Epoch 683/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9049 - accuracy: 0.7932
    Epoch 684/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9041 - accuracy: 0.7945
    Epoch 685/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8920 - accuracy: 0.7959
    Epoch 686/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9227 - accuracy: 0.7912
    Epoch 687/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8822 - accuracy: 0.7997
    Epoch 688/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8739 - accuracy: 0.7979
    Epoch 689/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8502 - accuracy: 0.8054
    Epoch 690/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8234 - accuracy: 0.8091
    Epoch 691/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8713 - accuracy: 0.8022
    Epoch 692/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8826 - accuracy: 0.7984
    Epoch 693/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8280 - accuracy: 0.8082
    Epoch 694/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9246 - accuracy: 0.7973
    Epoch 695/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9512 - accuracy: 0.7844
    Epoch 696/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9920 - accuracy: 0.7804
    Epoch 697/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9702 - accuracy: 0.7791
    Epoch 698/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0239 - accuracy: 0.7847
    Epoch 699/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.7271 - accuracy: 0.6585
    Epoch 700/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2763 - accuracy: 0.7041
    Epoch 701/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1345 - accuracy: 0.7338
    Epoch 702/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2150 - accuracy: 0.7347
    Epoch 703/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0815 - accuracy: 0.7427
    Epoch 704/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9798 - accuracy: 0.7678
    Epoch 705/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9790 - accuracy: 0.7729
    Epoch 706/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0097 - accuracy: 0.7661
    Epoch 707/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9343 - accuracy: 0.7796
    Epoch 708/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9046 - accuracy: 0.7863
    Epoch 709/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8848 - accuracy: 0.7923
    Epoch 710/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8923 - accuracy: 0.7934
    Epoch 711/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8700 - accuracy: 0.8005
    Epoch 712/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8902 - accuracy: 0.7978
    Epoch 713/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8349 - accuracy: 0.8070
    Epoch 714/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9207 - accuracy: 0.7926
    Epoch 715/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9117 - accuracy: 0.7920
    Epoch 716/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8705 - accuracy: 0.7972
    Epoch 717/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8340 - accuracy: 0.8100
    Epoch 718/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8510 - accuracy: 0.8047
    Epoch 719/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8337 - accuracy: 0.8091
    Epoch 720/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9619 - accuracy: 0.7892
    Epoch 721/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9849 - accuracy: 0.7778
    Epoch 722/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8959 - accuracy: 0.7930
    Epoch 723/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9426 - accuracy: 0.7969
    Epoch 724/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0828 - accuracy: 0.7617
    Epoch 725/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9413 - accuracy: 0.7837
    Epoch 726/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9583 - accuracy: 0.7831
    Epoch 727/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8995 - accuracy: 0.7929
    Epoch 728/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8712 - accuracy: 0.7981
    Epoch 729/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8373 - accuracy: 0.8051
    Epoch 730/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8240 - accuracy: 0.8111
    Epoch 731/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8897 - accuracy: 0.8051
    Epoch 732/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3336 - accuracy: 0.7362
    Epoch 733/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9554 - accuracy: 0.7777
    Epoch 734/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9202 - accuracy: 0.7887
    Epoch 735/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8712 - accuracy: 0.7970
    Epoch 736/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9246 - accuracy: 0.7995
    Epoch 737/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1569 - accuracy: 0.7502
    Epoch 738/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9765 - accuracy: 0.7752
    Epoch 739/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9183 - accuracy: 0.7834
    Epoch 740/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8785 - accuracy: 0.7956
    Epoch 741/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9080 - accuracy: 0.7955
    Epoch 742/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8986 - accuracy: 0.7961
    Epoch 743/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4933 - accuracy: 0.7327
    Epoch 744/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3755 - accuracy: 0.7035
    Epoch 745/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2379 - accuracy: 0.7237
    Epoch 746/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1409 - accuracy: 0.7427
    Epoch 747/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3340 - accuracy: 0.7062
    Epoch 748/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1384 - accuracy: 0.7319
    Epoch 749/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1430 - accuracy: 0.7391
    Epoch 750/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1527 - accuracy: 0.7334
    Epoch 751/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0621 - accuracy: 0.7540
    Epoch 752/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0572 - accuracy: 0.7616
    Epoch 753/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0359 - accuracy: 0.7631
    Epoch 754/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1057 - accuracy: 0.7525
    Epoch 755/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9678 - accuracy: 0.7759
    Epoch 756/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9219 - accuracy: 0.7866
    Epoch 757/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9077 - accuracy: 0.7892
    Epoch 758/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9464 - accuracy: 0.7862
    Epoch 759/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9467 - accuracy: 0.7877
    Epoch 760/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9223 - accuracy: 0.7886
    Epoch 761/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9050 - accuracy: 0.7947
    Epoch 762/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0867 - accuracy: 0.7718
    Epoch 763/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1631 - accuracy: 0.7529
    Epoch 764/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9989 - accuracy: 0.7760
    Epoch 765/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9713 - accuracy: 0.7763
    Epoch 766/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8701 - accuracy: 0.7987
    Epoch 767/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8556 - accuracy: 0.8011
    Epoch 768/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8395 - accuracy: 0.8048
    Epoch 769/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9381 - accuracy: 0.7905
    Epoch 770/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8842 - accuracy: 0.7959
    Epoch 771/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8852 - accuracy: 0.7984
    Epoch 772/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8926 - accuracy: 0.7961
    Epoch 773/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9656 - accuracy: 0.7877
    Epoch 774/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9519 - accuracy: 0.7832
    Epoch 775/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2405 - accuracy: 0.7552
    Epoch 776/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1135 - accuracy: 0.7481
    Epoch 777/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9740 - accuracy: 0.7753
    Epoch 778/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9958 - accuracy: 0.7770
    Epoch 779/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3955 - accuracy: 0.6995
    Epoch 780/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1402 - accuracy: 0.7344
    Epoch 781/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0445 - accuracy: 0.7553
    Epoch 782/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0123 - accuracy: 0.7625
    Epoch 783/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9257 - accuracy: 0.7825
    Epoch 784/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8960 - accuracy: 0.7903
    Epoch 785/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9216 - accuracy: 0.7880
    Epoch 786/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8968 - accuracy: 0.7907
    Epoch 787/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8722 - accuracy: 0.7973
    Epoch 788/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8734 - accuracy: 0.7977
    Epoch 789/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8373 - accuracy: 0.8050
    Epoch 790/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8580 - accuracy: 0.8043
    Epoch 791/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0420 - accuracy: 0.7773
    Epoch 792/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1903 - accuracy: 0.7488
    Epoch 793/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9598 - accuracy: 0.7803
    Epoch 794/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9709 - accuracy: 0.7813
    Epoch 795/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9625 - accuracy: 0.7855
    Epoch 796/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0391 - accuracy: 0.7634
    Epoch 797/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9821 - accuracy: 0.7654
    Epoch 798/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8904 - accuracy: 0.7933
    Epoch 799/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9479 - accuracy: 0.7828
    Epoch 800/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8823 - accuracy: 0.7985
    Epoch 801/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9952 - accuracy: 0.7779
    Epoch 802/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8828 - accuracy: 0.7961
    Epoch 803/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8571 - accuracy: 0.8016
    Epoch 804/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8801 - accuracy: 0.7975
    Epoch 805/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9865 - accuracy: 0.7828
    Epoch 806/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8641 - accuracy: 0.7990
    Epoch 807/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8431 - accuracy: 0.8042
    Epoch 808/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8164 - accuracy: 0.8133
    Epoch 809/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8580 - accuracy: 0.8034
    Epoch 810/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8556 - accuracy: 0.8043
    Epoch 811/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8105 - accuracy: 0.8132
    Epoch 812/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8188 - accuracy: 0.8131
    Epoch 813/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8743 - accuracy: 0.8045
    Epoch 814/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8169 - accuracy: 0.8118
    Epoch 815/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8162 - accuracy: 0.8122
    Epoch 816/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9775 - accuracy: 0.7970
    Epoch 817/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1625 - accuracy: 0.7476
    Epoch 818/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9001 - accuracy: 0.7884
    Epoch 819/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0641 - accuracy: 0.7750
    Epoch 820/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9919 - accuracy: 0.7805
    Epoch 821/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8723 - accuracy: 0.7966
    Epoch 822/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8727 - accuracy: 0.7993
    Epoch 823/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8410 - accuracy: 0.8045
    Epoch 824/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8529 - accuracy: 0.8023
    Epoch 825/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8341 - accuracy: 0.8083
    Epoch 826/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9137 - accuracy: 0.7988
    Epoch 827/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9608 - accuracy: 0.7817
    Epoch 828/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9194 - accuracy: 0.7886
    Epoch 829/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8665 - accuracy: 0.7997
    Epoch 830/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8182 - accuracy: 0.8090
    Epoch 831/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1394 - accuracy: 0.7702
    Epoch 832/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0445 - accuracy: 0.7593
    Epoch 833/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8828 - accuracy: 0.7958
    Epoch 834/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8717 - accuracy: 0.8010
    Epoch 835/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9231 - accuracy: 0.7897
    Epoch 836/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8679 - accuracy: 0.8015
    Epoch 837/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8268 - accuracy: 0.8068
    Epoch 838/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8002 - accuracy: 0.8139
    Epoch 839/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9197 - accuracy: 0.7936
    Epoch 840/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8586 - accuracy: 0.8045
    Epoch 841/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8872 - accuracy: 0.8001
    Epoch 842/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9480 - accuracy: 0.7880
    Epoch 843/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8407 - accuracy: 0.8071
    Epoch 844/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8959 - accuracy: 0.7958
    Epoch 845/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9136 - accuracy: 0.8030
    Epoch 846/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3654 - accuracy: 0.7312
    Epoch 847/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9542 - accuracy: 0.7736
    Epoch 848/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8931 - accuracy: 0.7954
    Epoch 849/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9024 - accuracy: 0.7917
    Epoch 850/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8484 - accuracy: 0.8030
    Epoch 851/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8366 - accuracy: 0.8073
    Epoch 852/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7956 - accuracy: 0.8160
    Epoch 853/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7977 - accuracy: 0.8145
    Epoch 854/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7813 - accuracy: 0.8189
    Epoch 855/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8216 - accuracy: 0.8140
    Epoch 856/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8065 - accuracy: 0.8155
    Epoch 857/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8809 - accuracy: 0.8070
    Epoch 858/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9590 - accuracy: 0.7869
    Epoch 859/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4137 - accuracy: 0.7263
    Epoch 860/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1905 - accuracy: 0.7239
    Epoch 861/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9372 - accuracy: 0.7784
    Epoch 862/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8642 - accuracy: 0.7995
    Epoch 863/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0099 - accuracy: 0.7742
    Epoch 864/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9111 - accuracy: 0.7892
    Epoch 865/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9627 - accuracy: 0.7844
    Epoch 866/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0485 - accuracy: 0.7692
    Epoch 867/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8705 - accuracy: 0.7963
    Epoch 868/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8686 - accuracy: 0.8027
    Epoch 869/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2169 - accuracy: 0.7592
    Epoch 870/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9245 - accuracy: 0.7824
    Epoch 871/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8516 - accuracy: 0.7998
    Epoch 872/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8870 - accuracy: 0.7965
    Epoch 873/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8894 - accuracy: 0.7972
    Epoch 874/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9023 - accuracy: 0.7970
    Epoch 875/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8343 - accuracy: 0.8038
    Epoch 876/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7990 - accuracy: 0.8163
    Epoch 877/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7932 - accuracy: 0.8166
    Epoch 878/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7868 - accuracy: 0.8186
    Epoch 879/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8100 - accuracy: 0.8151
    Epoch 880/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9045 - accuracy: 0.7995
    Epoch 881/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0873 - accuracy: 0.7680
    Epoch 882/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9736 - accuracy: 0.7814
    Epoch 883/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9541 - accuracy: 0.7814
    Epoch 884/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0217 - accuracy: 0.7801
    Epoch 885/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9308 - accuracy: 0.7855
    Epoch 886/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8508 - accuracy: 0.8008
    Epoch 887/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8538 - accuracy: 0.8041
    Epoch 888/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0323 - accuracy: 0.7780
    Epoch 889/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9270 - accuracy: 0.7916
    Epoch 890/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9093 - accuracy: 0.7983
    Epoch 891/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8741 - accuracy: 0.7987
    Epoch 892/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8255 - accuracy: 0.8096
    Epoch 893/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8758 - accuracy: 0.8002
    Epoch 894/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8500 - accuracy: 0.8091
    Epoch 895/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8584 - accuracy: 0.8020
    Epoch 896/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8141 - accuracy: 0.8132
    Epoch 897/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8022 - accuracy: 0.8133
    Epoch 898/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8021 - accuracy: 0.8176
    Epoch 899/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8365 - accuracy: 0.8168
    Epoch 900/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2734 - accuracy: 0.7424
    Epoch 901/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0808 - accuracy: 0.7677
    Epoch 902/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0666 - accuracy: 0.7576
    Epoch 903/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9183 - accuracy: 0.7851
    Epoch 904/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9344 - accuracy: 0.7893
    Epoch 905/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8420 - accuracy: 0.8044
    Epoch 906/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8565 - accuracy: 0.8035
    Epoch 907/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8163 - accuracy: 0.8106
    Epoch 908/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8415 - accuracy: 0.8120
    Epoch 909/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9172 - accuracy: 0.7945
    Epoch 910/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9378 - accuracy: 0.7995
    Epoch 911/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8810 - accuracy: 0.8012
    Epoch 912/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9438 - accuracy: 0.7889
    Epoch 913/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8654 - accuracy: 0.7998
    Epoch 914/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7946 - accuracy: 0.8150
    Epoch 915/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8942 - accuracy: 0.8016
    Epoch 916/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0284 - accuracy: 0.7775
    Epoch 917/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0847 - accuracy: 0.7601
    Epoch 918/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9627 - accuracy: 0.7829
    Epoch 919/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9210 - accuracy: 0.7849
    Epoch 920/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8590 - accuracy: 0.8026
    Epoch 921/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1166 - accuracy: 0.7579
    Epoch 922/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9503 - accuracy: 0.7808
    Epoch 923/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8363 - accuracy: 0.8058
    Epoch 924/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8042 - accuracy: 0.8103
    Epoch 925/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7775 - accuracy: 0.8193
    Epoch 926/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7711 - accuracy: 0.8213
    Epoch 927/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7840 - accuracy: 0.8189
    Epoch 928/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7837 - accuracy: 0.8228
    Epoch 929/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8342 - accuracy: 0.8078
    Epoch 930/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7883 - accuracy: 0.8186
    Epoch 931/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7936 - accuracy: 0.8193
    Epoch 932/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.4103 - accuracy: 0.7330
    Epoch 933/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1984 - accuracy: 0.7241
    Epoch 934/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9847 - accuracy: 0.7689
    Epoch 935/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8603 - accuracy: 0.7966
    Epoch 936/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8364 - accuracy: 0.8055
    Epoch 937/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8885 - accuracy: 0.7980
    Epoch 938/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8342 - accuracy: 0.8062
    Epoch 939/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8886 - accuracy: 0.8019
    Epoch 940/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9102 - accuracy: 0.8005
    Epoch 941/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3633 - accuracy: 0.7248
    Epoch 942/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0109 - accuracy: 0.7641
    Epoch 943/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8669 - accuracy: 0.7967
    Epoch 944/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8391 - accuracy: 0.8051
    Epoch 945/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8325 - accuracy: 0.8091
    Epoch 946/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9142 - accuracy: 0.7948
    Epoch 947/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8924 - accuracy: 0.8009
    Epoch 948/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8900 - accuracy: 0.7973
    Epoch 949/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8483 - accuracy: 0.8059
    Epoch 950/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7866 - accuracy: 0.8181
    Epoch 951/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2058 - accuracy: 0.7654
    Epoch 952/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2438 - accuracy: 0.7255
    Epoch 953/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0213 - accuracy: 0.7645
    Epoch 954/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9904 - accuracy: 0.7702
    Epoch 955/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9541 - accuracy: 0.7772
    Epoch 956/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8906 - accuracy: 0.7911
    Epoch 957/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8517 - accuracy: 0.7996
    Epoch 958/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8221 - accuracy: 0.8047
    Epoch 959/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8625 - accuracy: 0.8044
    Epoch 960/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8500 - accuracy: 0.8036
    Epoch 961/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9684 - accuracy: 0.7917
    Epoch 962/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2090 - accuracy: 0.7468
    Epoch 963/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1368 - accuracy: 0.7457
    Epoch 964/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9542 - accuracy: 0.7795
    Epoch 965/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9675 - accuracy: 0.7816
    Epoch 966/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1109 - accuracy: 0.7559
    Epoch 967/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9983 - accuracy: 0.7671
    Epoch 968/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8646 - accuracy: 0.7966
    Epoch 969/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8241 - accuracy: 0.8103
    Epoch 970/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8718 - accuracy: 0.8016
    Epoch 971/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8154 - accuracy: 0.8137
    Epoch 972/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9128 - accuracy: 0.7948
    Epoch 973/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8836 - accuracy: 0.7973
    Epoch 974/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8264 - accuracy: 0.8056
    Epoch 975/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8184 - accuracy: 0.8108
    Epoch 976/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8202 - accuracy: 0.8104
    Epoch 977/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8165 - accuracy: 0.8095
    Epoch 978/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7737 - accuracy: 0.8216
    Epoch 979/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9127 - accuracy: 0.8006
    Epoch 980/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8498 - accuracy: 0.8078
    Epoch 981/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8082 - accuracy: 0.8122
    Epoch 982/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0241 - accuracy: 0.7754
    Epoch 983/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0534 - accuracy: 0.7689
    Epoch 984/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9125 - accuracy: 0.7902
    Epoch 985/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8453 - accuracy: 0.8019
    Epoch 986/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8405 - accuracy: 0.8063
    Epoch 987/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8486 - accuracy: 0.8070
    Epoch 988/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8837 - accuracy: 0.8030
    Epoch 989/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0771 - accuracy: 0.7669
    Epoch 990/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8841 - accuracy: 0.7982
    Epoch 991/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8086 - accuracy: 0.8116
    Epoch 992/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8227 - accuracy: 0.8098
    Epoch 993/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7877 - accuracy: 0.8175
    Epoch 994/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8014 - accuracy: 0.8182
    Epoch 995/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7601 - accuracy: 0.8248
    Epoch 996/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7612 - accuracy: 0.8252
    Epoch 997/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7463 - accuracy: 0.8300
    Epoch 998/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7524 - accuracy: 0.8267
    Epoch 999/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7634 - accuracy: 0.8280
    Epoch 1000/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7815 - accuracy: 0.8228
    Epoch 1001/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7818 - accuracy: 0.8227
    Epoch 1002/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8048 - accuracy: 0.8183
    Epoch 1003/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1462 - accuracy: 0.7743
    Epoch 1004/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9655 - accuracy: 0.7796
    Epoch 1005/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8482 - accuracy: 0.8031
    Epoch 1006/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1205 - accuracy: 0.7781
    Epoch 1007/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1521 - accuracy: 0.7503
    Epoch 1008/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9773 - accuracy: 0.7694
    Epoch 1009/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8574 - accuracy: 0.8006
    Epoch 1010/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9236 - accuracy: 0.7891
    Epoch 1011/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8731 - accuracy: 0.7997
    Epoch 1012/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8601 - accuracy: 0.8071
    Epoch 1013/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2435 - accuracy: 0.7663
    Epoch 1014/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2635 - accuracy: 0.7133
    Epoch 1015/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9964 - accuracy: 0.7643
    Epoch 1016/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9236 - accuracy: 0.7876
    Epoch 1017/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0638 - accuracy: 0.7620
    Epoch 1018/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8783 - accuracy: 0.7942
    Epoch 1019/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8851 - accuracy: 0.7988
    Epoch 1020/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8611 - accuracy: 0.7977
    Epoch 1021/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8504 - accuracy: 0.8045
    Epoch 1022/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7826 - accuracy: 0.8167
    Epoch 1023/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7875 - accuracy: 0.8171
    Epoch 1024/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7792 - accuracy: 0.8207
    Epoch 1025/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7571 - accuracy: 0.8243
    Epoch 1026/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7609 - accuracy: 0.8262
    Epoch 1027/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7556 - accuracy: 0.8279
    Epoch 1028/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7749 - accuracy: 0.8229
    Epoch 1029/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9378 - accuracy: 0.8044
    Epoch 1030/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0512 - accuracy: 0.7720
    Epoch 1031/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8852 - accuracy: 0.7950
    Epoch 1032/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7833 - accuracy: 0.8147
    Epoch 1033/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8568 - accuracy: 0.8049
    Epoch 1034/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8010 - accuracy: 0.8163
    Epoch 1035/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7780 - accuracy: 0.8218
    Epoch 1036/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1306 - accuracy: 0.7748
    Epoch 1037/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0028 - accuracy: 0.7764
    Epoch 1038/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8438 - accuracy: 0.8062
    Epoch 1039/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8631 - accuracy: 0.8053
    Epoch 1040/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8032 - accuracy: 0.8158
    Epoch 1041/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8364 - accuracy: 0.8109
    Epoch 1042/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7981 - accuracy: 0.8180
    Epoch 1043/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7728 - accuracy: 0.8209
    Epoch 1044/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7560 - accuracy: 0.8241
    Epoch 1045/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7394 - accuracy: 0.8306
    Epoch 1046/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7683 - accuracy: 0.8263
    Epoch 1047/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7320 - accuracy: 0.8313
    Epoch 1048/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7368 - accuracy: 0.8297
    Epoch 1049/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7388 - accuracy: 0.8322
    Epoch 1050/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7558 - accuracy: 0.8279
    Epoch 1051/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7656 - accuracy: 0.8274
    Epoch 1052/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3005 - accuracy: 0.7763
    Epoch 1053/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1379 - accuracy: 0.7491
    Epoch 1054/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0164 - accuracy: 0.7711
    Epoch 1055/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0093 - accuracy: 0.7665
    Epoch 1056/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0273 - accuracy: 0.7734
    Epoch 1057/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9174 - accuracy: 0.7897
    Epoch 1058/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8939 - accuracy: 0.7938
    Epoch 1059/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8326 - accuracy: 0.8084
    Epoch 1060/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8155 - accuracy: 0.8130
    Epoch 1061/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7991 - accuracy: 0.8179
    Epoch 1062/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7648 - accuracy: 0.8226
    Epoch 1063/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8359 - accuracy: 0.8179
    Epoch 1064/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2318 - accuracy: 0.7522
    Epoch 1065/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0774 - accuracy: 0.7530
    Epoch 1066/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8953 - accuracy: 0.7908
    Epoch 1067/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8149 - accuracy: 0.8087
    Epoch 1068/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7908 - accuracy: 0.8161
    Epoch 1069/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8892 - accuracy: 0.7977
    Epoch 1070/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7739 - accuracy: 0.8205
    Epoch 1071/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7509 - accuracy: 0.8264
    Epoch 1072/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8104 - accuracy: 0.8172
    Epoch 1073/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8069 - accuracy: 0.8136
    Epoch 1074/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7314 - accuracy: 0.8298
    Epoch 1075/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7776 - accuracy: 0.8252
    Epoch 1076/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7699 - accuracy: 0.8257
    Epoch 1077/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9268 - accuracy: 0.8014
    Epoch 1078/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9559 - accuracy: 0.7928
    Epoch 1079/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9681 - accuracy: 0.7924
    Epoch 1080/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0659 - accuracy: 0.7661
    Epoch 1081/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8404 - accuracy: 0.8055
    Epoch 1082/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8692 - accuracy: 0.8029
    Epoch 1083/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7612 - accuracy: 0.8239
    Epoch 1084/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7419 - accuracy: 0.8281
    Epoch 1085/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7359 - accuracy: 0.8291
    Epoch 1086/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7122 - accuracy: 0.8337
    Epoch 1087/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7383 - accuracy: 0.8331
    Epoch 1088/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9690 - accuracy: 0.7998
    Epoch 1089/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9724 - accuracy: 0.7910
    Epoch 1090/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8566 - accuracy: 0.8077
    Epoch 1091/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9836 - accuracy: 0.7809
    Epoch 1092/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8660 - accuracy: 0.8016
    Epoch 1093/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8215 - accuracy: 0.8123
    Epoch 1094/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7627 - accuracy: 0.8234
    Epoch 1095/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7421 - accuracy: 0.8270
    Epoch 1096/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7384 - accuracy: 0.8311
    Epoch 1097/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7815 - accuracy: 0.8241
    Epoch 1098/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0300 - accuracy: 0.7998
    Epoch 1099/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2639 - accuracy: 0.7512
    Epoch 1100/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9264 - accuracy: 0.7897
    Epoch 1101/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9152 - accuracy: 0.7969
    Epoch 1102/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9279 - accuracy: 0.7926
    Epoch 1103/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8331 - accuracy: 0.8080
    Epoch 1104/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0487 - accuracy: 0.7778
    Epoch 1105/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9469 - accuracy: 0.7833
    Epoch 1106/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8105 - accuracy: 0.8106
    Epoch 1107/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7807 - accuracy: 0.8209
    Epoch 1108/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8357 - accuracy: 0.8087
    Epoch 1109/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7935 - accuracy: 0.8173
    Epoch 1110/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9883 - accuracy: 0.7889
    Epoch 1111/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8009 - accuracy: 0.8119
    Epoch 1112/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7967 - accuracy: 0.8159
    Epoch 1113/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7668 - accuracy: 0.8251
    Epoch 1114/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7962 - accuracy: 0.8206
    Epoch 1115/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7836 - accuracy: 0.8256
    Epoch 1116/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7586 - accuracy: 0.8269
    Epoch 1117/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0378 - accuracy: 0.7970
    Epoch 1118/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0133 - accuracy: 0.7781
    Epoch 1119/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9079 - accuracy: 0.7940
    Epoch 1120/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8750 - accuracy: 0.7978
    Epoch 1121/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8101 - accuracy: 0.8130
    Epoch 1122/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7983 - accuracy: 0.8141
    Epoch 1123/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7754 - accuracy: 0.8230
    Epoch 1124/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8402 - accuracy: 0.8076
    Epoch 1125/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9814 - accuracy: 0.7852
    Epoch 1126/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0337 - accuracy: 0.7731
    Epoch 1127/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8857 - accuracy: 0.7988
    Epoch 1128/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7961 - accuracy: 0.8161
    Epoch 1129/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7891 - accuracy: 0.8199
    Epoch 1130/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7518 - accuracy: 0.8251
    Epoch 1131/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7600 - accuracy: 0.8241
    Epoch 1132/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7574 - accuracy: 0.8255
    Epoch 1133/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7997 - accuracy: 0.8173
    Epoch 1134/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7696 - accuracy: 0.8232
    Epoch 1135/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7230 - accuracy: 0.8316
    Epoch 1136/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7851 - accuracy: 0.8225
    Epoch 1137/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9041 - accuracy: 0.8110
    Epoch 1138/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.6307 - accuracy: 0.6864
    Epoch 1139/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1347 - accuracy: 0.7375
    Epoch 1140/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0088 - accuracy: 0.7684
    Epoch 1141/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8719 - accuracy: 0.7974
    Epoch 1142/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8495 - accuracy: 0.8098
    Epoch 1143/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8687 - accuracy: 0.7998
    Epoch 1144/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7763 - accuracy: 0.8160
    Epoch 1145/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8107 - accuracy: 0.8140
    Epoch 1146/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1590 - accuracy: 0.7543
    Epoch 1147/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9261 - accuracy: 0.7941
    Epoch 1148/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9829 - accuracy: 0.7889
    Epoch 1149/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8683 - accuracy: 0.8049
    Epoch 1150/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7830 - accuracy: 0.8184
    Epoch 1151/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7591 - accuracy: 0.8268
    Epoch 1152/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7999 - accuracy: 0.8212
    Epoch 1153/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7679 - accuracy: 0.8220
    Epoch 1154/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7558 - accuracy: 0.8253
    Epoch 1155/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8852 - accuracy: 0.8098
    Epoch 1156/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9157 - accuracy: 0.7963
    Epoch 1157/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8708 - accuracy: 0.8016
    Epoch 1158/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8089 - accuracy: 0.8155
    Epoch 1159/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8180 - accuracy: 0.8170
    Epoch 1160/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8655 - accuracy: 0.8107
    Epoch 1161/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0763 - accuracy: 0.7726
    Epoch 1162/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8509 - accuracy: 0.8043
    Epoch 1163/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0896 - accuracy: 0.7730
    Epoch 1164/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9130 - accuracy: 0.7873
    Epoch 1165/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7991 - accuracy: 0.8096
    Epoch 1166/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9903 - accuracy: 0.7889
    Epoch 1167/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9040 - accuracy: 0.7934
    Epoch 1168/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7944 - accuracy: 0.8135
    Epoch 1169/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7592 - accuracy: 0.8220
    Epoch 1170/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8359 - accuracy: 0.8078
    Epoch 1171/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7681 - accuracy: 0.8213
    Epoch 1172/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7926 - accuracy: 0.8193
    Epoch 1173/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7749 - accuracy: 0.8211
    Epoch 1174/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8200 - accuracy: 0.8163
    Epoch 1175/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8472 - accuracy: 0.8126
    Epoch 1176/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0204 - accuracy: 0.7867
    Epoch 1177/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9049 - accuracy: 0.7954
    Epoch 1178/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9082 - accuracy: 0.7970
    Epoch 1179/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9921 - accuracy: 0.7884
    Epoch 1180/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0902 - accuracy: 0.7677
    Epoch 1181/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9100 - accuracy: 0.7914
    Epoch 1182/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9040 - accuracy: 0.7997
    Epoch 1183/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8107 - accuracy: 0.8118
    Epoch 1184/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7718 - accuracy: 0.8216
    Epoch 1185/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7821 - accuracy: 0.8201
    Epoch 1186/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8465 - accuracy: 0.8101
    Epoch 1187/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7612 - accuracy: 0.8227
    Epoch 1188/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7526 - accuracy: 0.8248
    Epoch 1189/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7317 - accuracy: 0.8294
    Epoch 1190/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7884 - accuracy: 0.8233
    Epoch 1191/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7833 - accuracy: 0.8238
    Epoch 1192/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8314 - accuracy: 0.8130
    Epoch 1193/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7739 - accuracy: 0.8234
    Epoch 1194/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8201 - accuracy: 0.8188
    Epoch 1195/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7391 - accuracy: 0.8301
    Epoch 1196/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7439 - accuracy: 0.8313
    Epoch 1197/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8037 - accuracy: 0.8194
    Epoch 1198/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8160 - accuracy: 0.8175
    Epoch 1199/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8625 - accuracy: 0.8072
    Epoch 1200/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1559 - accuracy: 0.7645
    Epoch 1201/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9334 - accuracy: 0.7915
    Epoch 1202/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8201 - accuracy: 0.8099
    Epoch 1203/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7789 - accuracy: 0.8194
    Epoch 1204/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7611 - accuracy: 0.8252
    Epoch 1205/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7972 - accuracy: 0.8199
    Epoch 1206/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7523 - accuracy: 0.8256
    Epoch 1207/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7591 - accuracy: 0.8267
    Epoch 1208/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7086 - accuracy: 0.8323
    Epoch 1209/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7134 - accuracy: 0.8364
    Epoch 1210/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7942 - accuracy: 0.8243
    Epoch 1211/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7498 - accuracy: 0.8312
    Epoch 1212/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7180 - accuracy: 0.8353
    Epoch 1213/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8239 - accuracy: 0.8208
    Epoch 1214/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8552 - accuracy: 0.8133
    Epoch 1215/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7702 - accuracy: 0.8243
    Epoch 1216/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7445 - accuracy: 0.8303
    Epoch 1217/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7325 - accuracy: 0.8295
    Epoch 1218/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7528 - accuracy: 0.8297
    Epoch 1219/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7349 - accuracy: 0.8345
    Epoch 1220/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8134 - accuracy: 0.8221
    Epoch 1221/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7362 - accuracy: 0.8298
    Epoch 1222/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8023 - accuracy: 0.8214
    Epoch 1223/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0438 - accuracy: 0.7836
    Epoch 1224/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9797 - accuracy: 0.7916
    Epoch 1225/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0809 - accuracy: 0.7794
    Epoch 1226/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9569 - accuracy: 0.7855
    Epoch 1227/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8487 - accuracy: 0.8023
    Epoch 1228/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2399 - accuracy: 0.7398
    Epoch 1229/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8895 - accuracy: 0.7872
    Epoch 1230/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7749 - accuracy: 0.8194
    Epoch 1231/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7635 - accuracy: 0.8218
    Epoch 1232/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7526 - accuracy: 0.8277
    Epoch 1233/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7555 - accuracy: 0.8247
    Epoch 1234/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7542 - accuracy: 0.8277
    Epoch 1235/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8091 - accuracy: 0.8210
    Epoch 1236/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.3029 - accuracy: 0.7418
    Epoch 1237/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8945 - accuracy: 0.7925
    Epoch 1238/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7910 - accuracy: 0.8127
    Epoch 1239/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7416 - accuracy: 0.8266
    Epoch 1240/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7207 - accuracy: 0.8305
    Epoch 1241/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7521 - accuracy: 0.8289
    Epoch 1242/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8478 - accuracy: 0.8107
    Epoch 1243/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7406 - accuracy: 0.8268
    Epoch 1244/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8316 - accuracy: 0.8209
    Epoch 1245/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8057 - accuracy: 0.8164
    Epoch 1246/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8289 - accuracy: 0.8163
    Epoch 1247/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8403 - accuracy: 0.8093
    Epoch 1248/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7621 - accuracy: 0.8230
    Epoch 1249/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8364 - accuracy: 0.8139
    Epoch 1250/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7533 - accuracy: 0.8254
    Epoch 1251/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7295 - accuracy: 0.8323
    Epoch 1252/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7384 - accuracy: 0.8324
    Epoch 1253/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.6886 - accuracy: 0.8393
    Epoch 1254/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7018 - accuracy: 0.8377
    Epoch 1255/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7995 - accuracy: 0.8239
    Epoch 1256/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.0334 - accuracy: 0.7855
    Epoch 1257/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8707 - accuracy: 0.8055
    Epoch 1258/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8083 - accuracy: 0.8141
    Epoch 1259/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8473 - accuracy: 0.8123
    Epoch 1260/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.1554 - accuracy: 0.7675
    Epoch 1261/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9038 - accuracy: 0.7925
    Epoch 1262/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7770 - accuracy: 0.8170
    Epoch 1263/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7442 - accuracy: 0.8273
    Epoch 1264/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8349 - accuracy: 0.8159
    Epoch 1265/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9615 - accuracy: 0.7860
    Epoch 1266/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8480 - accuracy: 0.8031
    Epoch 1267/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7783 - accuracy: 0.8196
    Epoch 1268/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7339 - accuracy: 0.8314
    Epoch 1269/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7190 - accuracy: 0.8341
    Epoch 1270/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7101 - accuracy: 0.8358
    Epoch 1271/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7065 - accuracy: 0.8388
    Epoch 1272/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7505 - accuracy: 0.8314
    Epoch 1273/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.7410 - accuracy: 0.8328
    Epoch 1274/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2788 - accuracy: 0.7575
    Epoch 1275/1280
    400/400 [==============================] - 1s 2ms/step - loss: 1.2029 - accuracy: 0.7481
    Epoch 1276/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8782 - accuracy: 0.7957
    Epoch 1277/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.9911 - accuracy: 0.7767
    Epoch 1278/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8869 - accuracy: 0.7958
    Epoch 1279/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8043 - accuracy: 0.8136
    Epoch 1280/1280
    400/400 [==============================] - 1s 2ms/step - loss: 0.8730 - accuracy: 0.8041
    model entrenat!



```python
plt.xlabel("Època")
plt.ylabel("Magnitud de pérdua")
plt.plot(historial.history["loss"])
```




    [<matplotlib.lines.Line2D at 0x7fb1b81ab8e0>]




    
![png](output_22_1.png)
    



```python
model.save('model_exportat2-07379.h5')
model.summary()
target
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     normalization (Normalizatio  (None, 32)               65        
     n)                                                              
                                                                     
     dense (Dense)               (None, 64)                2112      
                                                                     
     dense_1 (Dense)             (None, 32)                2080      
                                                                     
     dense_2 (Dense)             (None, 16)                528       
                                                                     
     dense_3 (Dense)             (None, 2)                 34        
                                                                     
    =================================================================
    Total params: 4,819
    Trainable params: 4,754
    Non-trainable params: 65
    _________________________________________________________________





    0     1.0
    1     1.0
    2     1.0
    3     1.0
    4     1.0
         ... 
    27    0.0
    28    0.0
    29    0.0
    30    0.0
    31    0.0
    Name: target, Length: 12800, dtype: float32




```python
tf.keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)

```




    
![png](output_24_0.png)
    




```python
'''numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

for row in numeric_dataset.take(3):
  print(row)
'''
```




    'numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))\n\nfor row in numeric_dataset.take(3):\n  print(row)\n'




```python
'''Anem a carregar el dataframe de test en memòria'''
directory = './salida/'
hdf_dirname =directory+'CaP-PBH-40-test.h5'
df1=pd.read_hdf(hdf_dirname, '/df1')
df1.head
#df1.sample(frac=1)
numeric_feature_test = df1[numeric_feature_names]
numeric_feature_test.head()
tensor_test=tf.convert_to_tensor(numeric_features)

```


```python
#tensor_test=tf.random.shuffle(
#    tensor_test, seed=None, name=None
#)

prediccio=model.predict(tensor_test)
```

    400/400 [==============================] - 1s 1ms/step



```python
print(prediccio)
```

    [[1.4526421e-02 9.8547351e-01]
     [3.2055404e-03 9.9679458e-01]
     [4.7261608e-04 9.9952739e-01]
     ...
     [9.9999112e-01 8.8489633e-06]
     [9.9999994e-01 3.7920486e-08]
     [9.9813700e-01 1.8629584e-03]]



```python
df1=pd.DataFrame(prediccio)
df1=df1[1].round()
display(df1) 
type(prediccio) #1 Càncer 0 Hiperplàssia 
```


    0        1.0
    1        1.0
    2        1.0
    3        1.0
    4        1.0
            ... 
    12795    0.0
    12796    0.0
    12797    0.0
    12798    0.0
    12799    0.0
    Name: 1, Length: 12800, dtype: float32





    numpy.ndarray




```python

df1.head
```




    <bound method NDFrame.head of 0        1.0
    1        1.0
    2        1.0
    3        1.0
    4        1.0
            ... 
    12795    0.0
    12796    0.0
    12797    0.0
    12798    0.0
    12799    0.0
    Name: 1, Length: 12800, dtype: float32>




```python
#np.savetxt("target.csv", target, delimiter=",")
```


```python
#np.savetxt("prediccio.csv", df1, delimiter=",")
```


```python
from sklearn.metrics import confusion_matrix
#confusion_matrix(df1, target)
df1.shape
tensor_target.shape

tensor_pre=tf.convert_to_tensor(df1)
def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)
tensor_pre = replacenan(tensor_pre)
#matrix = tf.math.confusion_matrix(df1, tensor_target)
```


```python
cnf_matrix =confusion_matrix(tensor_target,tensor_pre)
```


```python
sns.heatmap(cnf_matrix.T, square=True, annot=True, fmt='d', cbar=False, cmap='gray')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Confusion Matrix')
plt.xticks([0.5,1.5],['HBP', 'CaP'], ha='center')
plt.yticks([0.5,1.5],['HBP', 'CaP'], va='center')
plt.savefig("./conf2.tiff")
plt.show()
```


    
![png](output_35_0.png)
    



```python
#plt.savefig("./conf2.svg") #No se puede tras show
!pwd
```

    /home/talens/ml4bda/RedNeuronal



```python
from sklearn.metrics import classification_report
classification_report(tensor_target,tensor_pre)
```




    '              precision    recall  f1-score   support\n\n         0.0       0.91      0.66      0.77      6400\n         1.0       0.73      0.94      0.82      6400\n\n    accuracy                           0.80     12800\n   macro avg       0.82      0.80      0.79     12800\nweighted avg       0.82      0.80      0.79     12800\n'




```python
#df = pd.DataFrame(classification_report(tensor_pre, tensor_target, digits=6, output_dict=True)).T
df = pd.DataFrame(classification_report(tensor_pre, tensor_target,
                                        labels=None,
                                        target_names=['HBP','CaP'],
                                        sample_weight=None,
                                        digits=4,
                                        output_dict=True,
                                        zero_division='warn')).T

df['support'] = df.support.apply(int)

df.style.background_gradient(cmap='gray',subset=pd.IndexSlice['0':'9', :'f1-score'])

import seaborn as sns
sns.heatmap(df, annot=True,cmap='gray')
plt.tight_layout() 
plt.savefig("./report2.tiff")
```


    
![png](output_38_0.png)
    



```python
import numpy as np
from sklearn.metrics import roc_curve

y_true=tensor_target
y_score=tensor_pre

def sensivity_specifity_cutoff(y_true, y_score):
    '''Find data-driven cut-off for classification
    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    References
    ----------
    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

sensivity_specifity_cutoff(y_true, y_score)
```




    1.0




```python
# Exportar el modelo a js crea dos firxers; un .bin i un .json
!tensorflowjs_converter --input_format keras model_exportat-2.h5 salida

```

    2022-11-27 09:48:46.624147: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2022-11-27 09:48:46.655005: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2022-11-27 09:48:47.392884: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
    2022-11-27 09:48:47.392974: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
    2022-11-27 09:48:47.392990: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
    Traceback (most recent call last):
      File "/home/talens/.local/bin/tensorflowjs_converter", line 8, in <module>
        sys.exit(pip_main())
      File "/home/talens/.local/lib/python3.8/site-packages/tensorflowjs/converters/converter.py", line 827, in pip_main
        main([' '.join(sys.argv[1:])])
      File "/home/talens/.local/lib/python3.8/site-packages/tensorflowjs/converters/converter.py", line 831, in main
        convert(argv[0].split(' '))
      File "/home/talens/.local/lib/python3.8/site-packages/tensorflowjs/converters/converter.py", line 817, in convert
        _dispatch_converter(input_format, output_format, args, quantization_dtype_map,
      File "/home/talens/.local/lib/python3.8/site-packages/tensorflowjs/converters/converter.py", line 500, in _dispatch_converter
        dispatch_keras_h5_to_tfjs_layers_model_conversion(
      File "/home/talens/.local/lib/python3.8/site-packages/tensorflowjs/converters/converter.py", line 76, in dispatch_keras_h5_to_tfjs_layers_model_conversion
        raise ValueError('Nonexistent path to HDF5 file: %s' % h5_path)
    ValueError: Nonexistent path to HDF5 file: model_exportat-2.h5



```python
from sklearn.metrics import precision_recall_fscore_support
y_true=tensor_target
y_pred=tensor_pre
precision_recall_fscore_support(y_true, y_pred,average='binary' )
```




    (0.7339786790834456, 0.9359375, 0.8227456905432318, None)




```python
# amb la el comando netron es pot visualitzar la xarxa neuronal
```


```python
!netron ./salida/model.json #en la terminal real
```

    Serving './salida/model.json' at http://localhost:8080



```python
#  ssh -L 8098:localhost:PORT-SERVING-AT talens@158.42.145.69
# al teu PC vas a http://localhost:8098
```
