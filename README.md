# Hate Speech Detection using Transformers

This project demonstrates hate speech detection in tweets using advanced NLP techniques and deep learning (PyTorch). The core model is a Transformer-based classifier.

## Project Workflow

1. **Data Download**  
   - Dataset: [Kaggle - Twitter Hate Speech](https://www.kaggle.com/vkrahul/twitter-hate-speech?select=train_E6oV3lV.csv)

2. **Data Preparation & Preprocessing**  
   - Clean tweets using regex and BeautifulSoup.
   - Normalize slang/local words.
   - Remove irrelevant tokens (e.g., 'user').

3. **Data Visualization**  
   - Word clouds for each class.
   - Label distribution analysis.

4. **Data Transformation**  
   - Tokenization and vocabulary building using torchtext and spaCy.
   - Train/validation split.

5. **Model Development**  
   - Custom Transformer architecture implemented in PyTorch.

6. **Training & Validation**  
   - Model training loop with accuracy tracking.
   - Evaluation using precision, recall, F1-score, and confusion matrix.

7. **Inference & Deployment**  
   - Model inference on new tweets.
   - Results saved to CSV.

## Requirements

- Python 3.8+
- PyTorch
- torchtext
- spaCy (`en_core_web_sm`)
- pandas, numpy, matplotlib, seaborn, wordcloud
- opendatasets

## Usage

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/Hate-Speech-Detection.git
   cd Hate-Speech-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Download the dataset**
   - Go to [Kaggle - Twitter Hate Speech](https://www.kaggle.com/vkrahul/twitter-hate-speech?select=train_E6oV3lV.csv) and download the CSV file.
   - Place the dataset in the project folder.

4. **Run the notebook**
   - Open `HateSpeech.ipynb` in Jupyter or VS Code.
   - Execute the cells step by step to preprocess data, train the model, and evaluate results.

5. **Inference**
   - After training, use the provided inference code in the notebook to classify new tweets.
   - Save results to CSV as shown in the notebook.

## References

- ["Attention Is All You Need"](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Peter Bloem's Transformer Blog](http://peterbloem.nl/blog/transformers)
- [Harvard NLP - The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

## Notes

- The model is trained on a highly imbalanced dataset; undersampling is used.
- For best results, update the slang dictionary and