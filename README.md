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


     #Detect Language
     
     $ pip3 install scikit-learn
     
     The predicted language for "Bonjour tout le monde" is: French
     The predicted language for "Hello everyone" is: English
     The predicted language for "Hola a todos" is: Spanish
     The predicted language for "Hallo zusammen" is: German
     The predicted language for "Ciao a tutti" is: Italian
     The predicted language for "Привет всем" is: Russian
     The predicted language for "مرحبا بالجميع" is: Arabic
     The predicted language for "नमस्ते सब लोग" is: Hindi
