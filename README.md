# shuffle_attention_for_SER
Speech emotion recogntion with shuffle attention

The existing speech emotion recognition models mostly extract information from either spatial attention or channel attention, which cannot fully extract information from the spectrogram. The proposed model extracts features from both channel and spatial aspects and introduces residual connections to deepen the network and enhance the accuracy of emotion prediction.The weighted accuracy (WA) and unweighted accuracy (UA) achieved by our model are 82.21% and 80.09% respectively, surpassing the baseline model(https://github.com/lessonxmk/Optimized_attention_for_SER)by approximately 2.87% and 2.55%. These results indicate a highly competitive performance of our proposed approach.

The steps to build this project are as follows:
1. Use handleIEMOCAP.py to rename the iemocap data to something that is easy to work with.
2. Use process_dataset.py to divide IEMOCAP data set to train set and test set, and generate CSV files.
3. Use shuffle_attention.py to train and save the models.
