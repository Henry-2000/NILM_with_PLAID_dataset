# NILM with PLAID dataset
In this repository are available codes for implementation of electrical loads classification and event detection in residential environments using PLAID dataset. But it can also be adapted to work with any high frequency dataset that offers voltage and current signals from individual and/or aggregated measurements of household appliances, as long as basic editting of data handling (CSV/Metadata files and directories) and parameters (like network and sample frequency) are made.

PLAID dataset is available in [here](https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619/2) (access date: 21 Mar 2022). Only needs submetered/aggregated and metadata files. Extract/save them on the same folder directory of codes.

Related work: https://repositorio.ifes.edu.br/bitstream/handle/123456789/1886/TCC_Rede_Neural_Convolucional_Cargas_Filtro.pdf?sequence=1&isAllowed=y (in portuguese)

Files content:
* [main.py](https://github.com/hsneto/iftex/blob/master/textuais/testes.tex#L10) - code that displays the sequence of options for data processing and graphs generation;
* [process_data.py](https://github.com/hsneto/iftex/blob/master/textuais/testes.tex#L38) - include functions that handle individual and aggregated data and creates dictionaries that helps organizing them;
* [steady_samples.py](https://github.com/hsneto/iftex/blob/master/textuais/testes.tex#L60) - include functions for selection of stationary signal intervals and RMS generation;
* [harmonics.py](https://github.com/hsneto/iftex/blob/master/textuais/testes.tex#L60) - include functions for harmonic filtering/selection and reconstruction of signals;
* [s_transform.py](https://github.com/hsneto/iftex/blob/master/textuais/testes.tex#L60) - include functions for Stockwell Transform implementation;
* [kalman_filter.py](https://github.com/hsneto/iftex/blob/master/textuais/testes.tex#L60) - include function for Kalman Filter implementation that returns residue signal (for event detection of aggregated data);
* [generate_graphs.py](https://github.com/hsneto/iftex/blob/master/textuais/testes.tex#L60) - include functions to generate graphs and V-I trajectories;
* [utilities.py](https://github.com/hsneto/iftex/blob/master/textuais/testes.tex#L60) - include functions for specific data handling;
* [cnn.py](https://github.com/hsneto/iftex/blob/master/textuais/testes.tex#L60) - include functions for convolutional neural network construction and V-I trajectories classification, once they are obtained.

Contact e-mail:
hcampaneli@gmail.com



