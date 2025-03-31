# language-detection-model
# Install packages
 
     $ pip3 install scikit-learn
 
 # Create model
 
     $ python3 train-model.py 
 
     Fitting 3 folds for each of 6 candidates, totalling 18 fits
     [CV] END ...classifier__alpha=0.1, tfidf__ngram_range=(1, 2); total time=   0.7s
     [CV] END ...classifier__alpha=0.1, tfidf__ngram_range=(1, 2); total time=   0.7s
     [CV] END ...classifier__alpha=0.1, tfidf__ngram_range=(1, 2); total time=   0.7s
     [CV] END ...classifier__alpha=0.5, tfidf__ngram_range=(1, 2); total time=   0.7s
     [CV] END ...classifier__alpha=0.5, tfidf__ngram_range=(1, 2); total time=   0.7s
     [CV] END ...classifier__alpha=0.5, tfidf__ngram_range=(1, 2); total time=   0.7s
     [CV] END ...classifier__alpha=1.0, tfidf__ngram_range=(1, 2); total time=   0.6s
     [CV] END ...classifier__alpha=1.0, tfidf__ngram_range=(1, 2); total time=   0.6s
     [CV] END ...classifier__alpha=1.0, tfidf__ngram_range=(1, 2); total time=   0.6s
     [CV] END ...classifier__alpha=0.5, tfidf__ngram_range=(2, 5); total time=   1.9s
     [CV] END ...classifier__alpha=0.5, tfidf__ngram_range=(2, 5); total time=   1.9s
     [CV] END ...classifier__alpha=0.1, tfidf__ngram_range=(2, 5); total time=   2.0s
     [CV] END ...classifier__alpha=0.1, tfidf__ngram_range=(2, 5); total time=   2.1s
     [CV] END ...classifier__alpha=0.1, tfidf__ngram_range=(2, 5); total time=   2.1s
     [CV] END ...classifier__alpha=0.5, tfidf__ngram_range=(2, 5); total time=   2.1s
     [CV] END ...classifier__alpha=1.0, tfidf__ngram_range=(2, 5); total time=   1.6s
     [CV] END ...classifier__alpha=1.0, tfidf__ngram_range=(2, 5); total time=   1.8s
     [CV] END ...classifier__alpha=1.0, tfidf__ngram_range=(2, 5); total time=   1.8s
     Accuracy: 98.69%
     Classification Report:
                   precision    recall  f1-score   support
     
           Arabic       1.00      1.00      1.00       106
           Danish       0.99      0.95      0.97        73
            Dutch       1.00      0.95      0.97       111
          English       0.95      1.00      0.97       291
           French       1.00      0.99      1.00       219
           German       1.00      0.97      0.98        93
            Greek       1.00      0.99      0.99        68
            Hindi       1.00      1.00      1.00        10
          Italian       0.99      0.99      0.99       145
          Kannada       1.00      1.00      1.00        66
        Malayalam       1.00      0.99      1.00       121
       Portugeese       0.98      0.99      0.98       144
          Russian       1.00      1.00      1.00       136
          Spanish       1.00      0.97      0.98       160
         Sweedish       0.96      0.98      0.97       133
            Tamil       1.00      1.00      1.00        87
          Turkish       1.00      1.00      1.00       105
     
         accuracy                           0.99      2068
        macro avg       0.99      0.99      0.99      2068
     weighted avg       0.99      0.99      0.99      2068
     
     Model training complete and saved as 'language_detection_model.pkl'
 
 # Detect language
     $ pip3 install scikit-learn
     
     The predicted language for "Bonjour tout le monde" is: French
     The predicted language for "Hello everyone" is: English
     The predicted language for "Hola a todos" is: Spanish
     The predicted language for "Hallo zusammen" is: German
     The predicted language for "Ciao a tutti" is: Italian
     The predicted language for "Привет всем" is: Russian
     The predicted language for "مرحبا بالجميع" is: Arabic
     The predicted language for "नमस्ते सब लोग" is: Hindi


     import joblib
 
 # Function to load the trained model and make predictions
 def load_and_predict(text):
     # Load the trained model from the saved file
     # The model was saved earlier using joblib during the training process
     model = joblib.load('./model/language_detection_model.pkl')
     
     # Make a prediction on the given text input
     # The model expects a list of texts, so we pass the text inside a list
     prediction = model.predict([text])
     return prediction[0]  # Return the predicted language label
 
 # Example usage with a variety of sample texts for different languages
 # This list contains text samples in different languages for testing the model's predictions
 sample_texts = [
     "Bonjour tout le monde",  # French
     "Hello everyone",        # English
     "Hola a todos",          # Spanish
     "Hallo zusammen",        # German
     "Ciao a tutti",          # Italian
     "Привет всем",           # Russian
     "مرحبا بالجميع",          # Arabic
     "नमस्ते सब लोग"            # Hindi
 ]
 
 # Iterate over each sample text, predict its language, and print the result
 for text in sample_texts:
     predicted_language = load_and_predict(text)  # Call the function to predict the language
     print(f'The predicted language for "{text}" is: {predicted_language}')  # Output the prediction


import pandas as pd
 from sklearn.model_selection import train_test_split, GridSearchCV
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.naive_bayes import MultinomialNB
 from sklearn.pipeline import Pipeline
 from sklearn.metrics import accuracy_score, classification_report
 import joblib
 
 # Load dataset (https://www.kaggle.com/datasets/basilb2s/language-detection)
 # Load the CSV file containing the language detection data into a pandas DataFrame
 dataset_path = './dataset/LanguageDetection.csv'
 data = pd.read_csv(dataset_path)
 
 # Separate the features (text) and labels (language)
 X = data['Text']  # Feature set containing text data
 y = data['Language']  # Labels representing the language of each text
 
 # Split the dataset into training and testing sets
 # 80% of the data will be used for training and 20% for testing
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
 # Create a pipeline with TF-IDF Vectorizer and Multinomial Naive Bayes model
 # Pipeline helps to chain together multiple steps, such as feature extraction and classification
 # TfidfVectorizer converts the text data into numerical features
 # MultinomialNB is used for classification based on these features
 pipeline = Pipeline([
     ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))),  # Use character-level n-grams to capture patterns within text
     ('classifier', MultinomialNB())  # Use Naive Bayes classifier for prediction
 ])
 
 # Use GridSearchCV to find the best parameters
 # GridSearchCV helps to perform hyperparameter tuning to find the optimal values for parameters
 param_grid = {
     'tfidf__ngram_range': [(1, 2), (2, 5)],  # Tune the n-gram range for TF-IDF
     'classifier__alpha': [0.1, 0.5, 1.0]  # Tune the smoothing parameter (alpha) for Naive Bayes
 }
 
 # Perform a grid search over the parameter grid using 3-fold cross-validation
 # n_jobs=-1 utilizes all available CPU cores, and verbose=2 shows progress output
 grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
 grid_search.fit(X_train, y_train)
 
 # Get the best model found by GridSearchCV
 model = grid_search.best_estimator_
 
 # Make predictions on the test set
 y_pred = model.predict(X_test)
 
 # Evaluate the model's performance on the test data
 # Print the accuracy and a classification report to show detailed performance metrics
 accuracy = accuracy_score(y_test, y_pred)
 print(f'Accuracy: {accuracy * 100:.2f}%')
 print('Classification Report:')
 print(classification_report(y_test, y_pred))
 
 # Save the trained model for future use
 # Save the model using joblib to a file, so that it can be loaded later without retraining
 joblib.dump(model, './model/language_detection_model.pkl')
 
 print("Model training complete and saved as 'language_detection_model.pkl'")
