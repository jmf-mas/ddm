# cuq-ae-redm: a Framework for uncertainty quantification on AEs-based methods for anomaly detection
> usage: run.py [-h] [-t] [-p] [-b [BATCH_SIZE]] [-l [LEARNING_RATE]] [-w [WEIGHT_DECAY]] [-m [MOMENTUM]] [-e [EPOCHS]]

> optional arguments:

>> -h, --help            show this help message and exit
>> 
>> -t, --is_train        training mode (default: False)
>> 
>>  -p, --to_process_data process data (default: False)
>> 
>>  -b [BATCH_SIZE], --batch_size [BATCH_SIZE]
>> 
>>  -l [LEARNING_RATE], --learning_rate [LEARNING_RATE]
>> 
>>  -w [WEIGHT_DECAY], --weight_decay [WEIGHT_DECAY]
>> 
>>  -m [MOMENTUM], --momentum [MOMENTUM]
>> 
>>  -e [EPOCHS], --epochs [EPOCHS]
>>

> Ensure that data sets are stored within a directory labeled data

> example of use
```
python run.py -t -p
```
