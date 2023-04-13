## Custom NER based resume parser using Spacy3

1. Downloading appropriate the skeleton of base [config_file](https://spacy.io/usage/training) as per the system requirement.

2. Clone the source data [repo](https://github.com/laxmimerit/CV-Parsing-using-Spacy-3.git) to train the custom `ner` model.

   ```bash
   git clone https://github.com/laxmimerit/CV-Parsing-using-Spacy-3.git
   ```

3. Generate the `config.cfg` file using the `base_config.cfg` i.e. used for coniguring the Spacy Model parameters.

   ```bash
   python -m spacy init fill-config base_config.cfg config.cfg
   ```

4. Parse and convert source training data into `*.spacy` format.

5. Train the blank `ner` based model on custom data using below command.

   ```bash
   python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy --gpu-id=0
   ```

   **Remark:** Here dev.spacy denotes the test data i.e. used for model evaluation

6. Evaluate model performance on Unseen data.
