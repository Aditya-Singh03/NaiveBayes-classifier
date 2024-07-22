# Naive Bayes Classifier for Sentiment Analysis (IMDB Movie Reviews)

This project implements a Naive Bayes classifier from scratch for sentiment analysis on movie reviews from the IMDB Large Movie Review Dataset. The classifier predicts whether a review is positive or negative.

> [!NOTE]
> Due to the size of the dataset, I have not uploaded the dataset in this repository, however, feel free to download the dataset from one of the sources below:
> - Kaggle:
>   - [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
>   - [https://www.kaggle.com/datasets/jcblaise/imdb-sentiments](https://www.kaggle.com/datasets/jcblaise/imdb-sentiments)
> - Stanford AI Lab (Original Source): 
>   - [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)
> - IMDb Datasets: 
>   - [https://datasets.imdbws.com/](https://datasets.imdbws.com/)



## Table of Contents

- [Naive Bayes Classifier for Sentiment Analysis (IMDB Movie Reviews)](#naive-bayes-classifier-for-sentiment-analysis-imdb-movie-reviews)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Implementation](#implementation)
  - [How to Use](#how-to-use)
  - [Experiments and Results](#experiments-and-results)
  - [Dependencies](#dependencies)
  - [Acknowledgements](#acknowledgements)

## Project Overview

This project aims to:

- **Understand and implement the Naive Bayes algorithm:** Demonstrate a fundamental text classification technique.
- **Build a classifier from scratch:** Avoid using existing libraries to gain a deeper understanding of the underlying concepts.
- **Explore impact of different parameters:** Analyze how choices like Laplace smoothing (alpha) affect model performance.
- **Evaluate on a real-world dataset:** Test the classifier on a standard sentiment analysis benchmark.

## Dataset

The IMDB Large Movie Review Dataset (Maas et al., 2011) is used for training and testing. The dataset contains 50,000 movie reviews, split evenly into positive and negative sentiments. This README assumes the dataset is organized into 'train' and 'test' folders with 'pos' and 'neg' subdirectories.

## Implementation

- **`NaiveBayes` class:**  The core implementation of the Naive Bayes classifier.
- **`word_counter` function:**  Calculates word frequencies within a given class.
- **`preprocess_text` function:** Cleans text data by removing stop words, punctuation, and numbers.
- **`load_training_set` and `load_test_set` functions:** Load specific percentages of data for experimentation.

Key features:

- Handles both product of probabilities and log of probabilities approaches.
- Optional Laplace smoothing for handling unseen words.
- Calculates accuracy, precision, and recall metrics.
- Generates confusion matrices to visualize performance.

## How to Use

1. ### **Clone:** 
    Clone this repository.
    ```bash
    git clone https://github.com/Aditya-Singh03/NaiveBayes-classifier.git
    ```
2. ### **Install:** 
   Ensure you have the required dependencies (see below).
    ```bash
    pip install -r requirements.txt
    ```
3. ### **How to run this program?** (its very simple and concise): 
   Execute `knnAndDt.py` to run the experiments and generate results.
   **Just type**
    ```bash
    python run.py
    ```

    in the terminal and you will be able to run both the models and create the their plots in one go. All the plots will also get saved in this very same folder. (Just make sure that when you try to run this file, you are in the same folder as the file.)
4. ### **Explore:** Examine the generated plots and analyze the findings.

    >[!NOTE]
    >Make sure to uncomment the `plt.savefig('Q1-1.png')` (such lines) to also save the figure


## Experiments and Results

The `run.py` script performs several experiments:

- **Comparison of Product vs. Log Probabilities:** Compares accuracy with and without the log transformation.
- **Impact of Laplace Smoothing (alpha):** Evaluates model performance with different alpha values.
- **Performance on Full and Half Training Sets:** Assesses how training set size affects accuracy.
- **Unbalanced Dataset:**  Examines the model's behavior when the training data is imbalanced.

**Results:**
   The results for each experiment are printed to the console, including accuracy, precision, recall, confusion matrices, and plots (when relevant). 

## Dependencies

- Python 3
- NLTK
- Seaborn
- Matplotlib
- NumPy

## Acknowledgements

- The IMDB Large Movie Review Dataset by Maas et al. (2011).