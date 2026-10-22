# Weather Detection

A machine learning model that detects the weather type of a day (sun, rain, drizzle, fog or snow) from precipitation, temperature, wind speed and month.

Built with Python and scikit-learn, and run on Google Colab.

## Dataset

Seattle weather, 2012 to 2015: 1,461 days, one row per day (`weather.csv`).

| Column | Meaning |
|---|---|
| date | Day of the observation |
| precipitation | Rainfall in mm |
| temp_max / temp_min | Highest and lowest temperature (C) |
| wind | Wind speed |
| weather | Target label: sun, rain, drizzle, fog, snow |

Class counts: rain 641, sun 640, fog 101, drizzle 53, snow 26. The data is imbalanced, so macro F1 is used along with accuracy.

## Approach

1. Add the month from the date, and three extra features: temperature range, average temperature and a wet/dry flag.
2. Split 80% train / 20% test (stratified, `random_state=42`).
3. Compare Logistic Regression, KNN, Decision Tree and Random Forest with 5-fold cross-validation on the training data.
4. Pick the best model by macro F1 (Random Forest), then test it once on the test set.

## Results (test set, 293 days)

| Metric | Score |
|---|---|
| Accuracy | 0.7986 |
| Macro F1 | 0.5169 |

| Class | Precision | Recall | F1 | Days |
|---|---|---|---|---|
| rain | 0.97 | 0.89 | 0.93 | 129 |
| sun | 0.78 | 0.86 | 0.82 | 128 |
| snow | 0.50 | 0.40 | 0.44 | 5 |
| fog | 0.27 | 0.30 | 0.29 | 20 |
| drizzle | 0.12 | 0.09 | 0.11 | 11 |

Rain and sun are detected well. Fog, drizzle and snow are weak because there are very few examples of them, and the four measurements do not separate them clearly (drizzle and fog look a lot like light rain and sun).

## Run on Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com), then File > Upload notebook > `weather_classifier_colab.ipynb`.
2. Runtime > Run all.
3. Upload `weather.csv` when the upload cell asks for it.

## Run on your computer

```bash
pip install -r requirements.txt
python weather_classifier.py
python weather_classifier.py --predict 10.9 10.6 2.8 4.5 1
```

The last command takes: precipitation, max temp, min temp, wind, month.

## Screenshots

### Code
![Code](screenshots/code.png)

### Model comparison and output
![Output](screenshots/output.png)

### Confusion matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

## Possible improvements

- More data for fog, drizzle and snow (or a longer time period)
- Add humidity, pressure and cloud cover, which separate fog and drizzle better
- Tune the Random Forest with GridSearchCV, or try gradient boosting
- Weather detection from photos with a CNN

## Files

- `weather_classifier_colab.ipynb`: Colab notebook
- `weather_classifier.py`: same program for running locally
- `weather.csv`: dataset
- `requirements.txt`: libraries need
