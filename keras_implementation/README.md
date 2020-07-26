# Keras Implementation

This folder contains complete implementation of **MyDenoiser** using the 
[Keras Functional API](https://keras.io/guides/functional_api/).

## Testing

To test inference with this model, you can use the [test.py](test.py) script by 
running the following from a command line located at *.../MyDenoiser/keras_implementation/*

```
python3 test.py \
--set_dir=<parent_directory_of_test_dataset> \
--set_names=<list_of_names_of_dataset_folders_to_test_on> \
--model_dir=<directory_of_MyDenoiser_model> \
--model_name=<name_of_hdf5_model_file> \
--save_result=1

```

After testing the model, a file named *results.txt* will be saved in [*data/results/<set_name>*](data/results), 
where the set_name corresponds to set_names[i] from the command-line arguments shown above.  

If you specify the save_result command-line argument when running test.py (as shown above),
the denoised images will also be saved in [*data/results/<set_name>*](data/results), alongside *results.txt*


## Training

To train this model, use the [train.py](train.py) script by running the following from 
a command line located at *.../MyDenoiser/keras_implementation/*

```



```
