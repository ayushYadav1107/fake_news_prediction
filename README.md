# Fake News Detection using Machine Learning

## Overview
This project is a machine learning-based Fake News Detection system that classifies news articles as either real or fake. It leverages Natural Language Processing (NLP) techniques to preprocess textual data and applies a Logistic Regression model for classification.

## Dataset
The dataset used in this project consists of news articles with the following attributes:
- **id**: Unique identifier for a news article
- **title**: Title of the news article
- **author**: Author of the article
- **text**: Full content of the article
- **label**: 1 for Fake News, 0 for Real News

## Technologies Used
- Python
- Pandas, NumPy (Data Processing)
- Natural Language Toolkit (NLTK) (Text Preprocessing)
- Scikit-Learn (Machine Learning Model)

## Steps Involved
1. **Data Preprocessing:**
   - Handling missing values by replacing them with empty strings.
   - Combining the `author` and `title` columns into a single `content` column.
   - Applying text cleaning, tokenization, stopword removal, and stemming.

2. **Feature Extraction:**
   - Converting textual data into numerical data using TF-IDF vectorization.

3. **Model Training:**
   - Splitting the dataset into training and testing sets (80%-20%).
   - Training a **Logistic Regression** model on the dataset.

4. **Evaluation:**
   - Achieved **98.66% accuracy** on training data.
   - Achieved **97.90% accuracy** on test data.

5. **Prediction System:**
   - Developed a predictive system that takes new textual input and classifies it as real or fake news.

## How to Run the Project
### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install numpy pandas nltk scikit-learn
```

### Run the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   ```
2. Navigate to the project folder:
   ```bash
   cd fake-news-detection
   ```
3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook
   ```
4. Execute all cells to train and test the model.

## Results
- The model effectively classifies fake and real news with high accuracy.
- Logistic Regression was chosen due to its strong performance in binary classification tasks.

## Future Enhancements
- Experimenting with deep learning models (e.g., LSTMs, Transformers) for better accuracy.
- Deploying the model as a web application.

## Contributing
Feel free to fork the repository, make enhancements, and submit pull requests. Suggestions and contributions are welcome!

