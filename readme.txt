Instructions to run the code:
1) Install virtual environment using pip: 
    a) "python3 -m pip install --user virtualenv" (mac)
    b) "py -m pip install --user virtualenv" (windows)
2) Create a new virtual environment
    a) "python3 -m venv env"
    b) "py -m venv env"
3) Activate the virtual environment
    a) "source env/bin/activate"
    b) ".\env\Scripts\activate"
4) Install the required dependencies, "pip install -r requirements.txt"
5) Run the python script for each dataset (1, 2, 3), "python3 analysisScript.py dataset[1,2,3].txt
6) Deactivate the environment using, "deactivate"

Explanation of the files generated during code generation:
1) 1_tokens_datasetX.txt --> List of all the tokens in the dataset X (1, 2, 3)
2) 2_tokens_without_sw_datasetX.txt --> List of all the tokens in the dataset X (1, 2, 3) without stopwords
3) 3_stemmed_words_datasetX.txt --> List of all the stemmed tokens in dataset 1_tokens_datasetX
4) 4_sentence_segmentation_datasetX.txt --> List of all segmented sentence in 1_tokens_datasetX
5) 5_phrases_with_improved_tokeniser_datasetX.txt --> Improved tokeniser that generates better tokens through phrase detection and extraction