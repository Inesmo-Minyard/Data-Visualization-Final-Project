# Beyond Popularity: What Makes Anime Successful?

## Project overview

This data visualisation project explores how anime characteristics are associated with audience reach, user ratings, and viewer loyalty.

The analysis examines anime atmosphere, release format, historical period, studio performance, hidden gems, and genre–theme combinations.

## Main research question

How are anime characteristics associated with popularity, user ratings, and audience loyalty, and how have these relationships changed over time?

## Dataset

The project uses a MyAnimeList-derived anime dataset containing information about:

- titles and release formats;
- user scores and number of ratings;
- members and favourites;
- genres and themes;
- studios;
- release years;
- source material;
- synopses.

## Analytical approach

The notebook includes ten multidimensional analytical questions covering:

1. Audience size and score across release formats
2. Audience loyalty across atmosphere categories
3. Typical popularity and rating by atmosphere
4. Rating consistency by atmosphere
5. Atmosphere production share across decades
6. Relative audience popularity across periods
7. Hidden gems
8. TV versus movie ratings
9. Studio performance within atmospheres
10. Genre–theme combinations

## Dashboard

The Streamlit dashboard provides interactive filters for:

- atmosphere;
- release format;
- release year;
- minimum number of user ratings.

Dashboard sections include:

- Overview
- Atmosphere Analysis
- Hidden Gems
- Studios
- Genre–Theme Combinations
- Methodology

## Technologies

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- JupyterLab

## Running the dashboard locally

Install the required libraries:

```bash
python3 -m pip install -r requirements.txt