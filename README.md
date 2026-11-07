# Lab 02 - Web Scraping, Feature Engineering and EDA

This project is prepared for Windows + Visual Studio Code.

## Important note about the supplied files
The DOCX file supplied with the name "Building an ETL Pipeline on a Bulk Kaggle Dataset" contains the same Web Scraping, Feature Engineering and EDA lab material as the PDF. It does not contain a separate Kaggle ETL assignment. This project follows the actual PDF and notebook content.

## Project tasks
1. **Task 2.1:** NumPy vectorised sum vs a normal Python loop.
2. **Task 2.2:** Web scraping of article URL, title and body text.
3. **Task 2.3:** spaCy linguistic features: tokens, sentences, entities and nouns.
4. **Task 2.3:** TextBlob sentiment: polarity and subjectivity.
5. **Task 2.4:** EDA histograms, pair plot and correlation matrix.
6. **Task 2.4:** TF-IDF top terms. Optional in-lab exercises are also included.

## Fast setup (Windows)
1. Install **Python 3.11 (64-bit)** from https://www.python.org/downloads/release/python-3119/
   During installation, tick **Add python.exe to PATH**.
2. Install Visual Studio Code and its **Python** extension by Microsoft.
3. Extract this ZIP to a normal folder, for example `Desktop\\Lab02_VSCode_Project`.
4. Open VS Code. Select **File > Open Folder** and choose `Lab02_VSCode_Project`.
5. In File Explorer, double-click `setup_windows.bat`. Wait for it to finish. It creates `.venv`, installs packages and downloads the spaCy English model.
6. In VS Code press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select `.venv\\Scripts\\python.exe`.
7. Open the terminal in VS Code (**Terminal > New Terminal**) and run:

   ```powershell
   python main.py
   ```

8. To run the two included in-lab exercises as well:

   ```powershell
   python main.py --exercises
   ```

9. To try live web scraping instead of the supplied CSV:

   ```powershell
   python main.py --scrape
   ```

   Live scraping needs an internet connection. If the website blocks requests or changes its HTML, the project falls back to the supplied CSV.

## In-lab exercises and home assignment
The notebook lists four in-lab exercises:
1. Add two more article URLs and inspect the HTML when a page fails to parse. The URL list is in `main.py`; the selectors are in `scrape_articles()`.
2. Add verb count and average sentence length. Implemented by `--exercises`.
3. Explain why KDE over five observations is not a population distribution. See `VIVA_PREPARATION.pdf`.
4. Compare TF-IDF with and without stop-word removal. Implemented by `--exercises`.

The home assignment asks for 20 articles from two blogs (10 each). The uploaded files do not supply those URLs or the second blog's confirmed HTML selectors. `home_assignment.py` is a ready scaffold, but you must add those 20 URLs and verify selectors before it can run. It then calculates mean sentiment polarity and noun density and creates a grouped chart.

## Run without typing commands
After setup, double-click:
- `run_lab.bat` for the main lab.
- `run_with_exercises.bat` for the main lab plus the included exercises.

## Input and output
- Input: `data/articles_raw.csv`
- Final engineered dataset: `outputs/articles.csv`
- Correlation values: `outputs/correlation_matrix.csv`
- TF-IDF terms: `outputs/tfidf_top_terms.csv`
- Graphs: `outputs/plots/`
- Exercise outputs: `outputs/exercise_*.csv`

The supplied CSV is used by default, so the main lab does not depend on live website availability.

## If something fails
- **`python` is not recognized:** reinstall Python 3.11 and tick Add Python to PATH, then reopen VS Code.
- **spaCy model missing:** activate `.venv` in the VS Code terminal and run `python -m spacy download en_core_web_sm`.
- **Wrong interpreter:** select `.venv\\Scripts\\python.exe` using Python: Select Interpreter.
- **Package missing:** run `python -m pip install -r requirements.txt` in the activated environment.
- **CSV missing:** confirm `data/articles_raw.csv` exists. Do not rename its columns; expected columns are `url`, `title`, `text`.

## Viva preparation
Use `VIVA_PREPARATION.pdf` for concise definitions, workflow, likely questions and answers.
