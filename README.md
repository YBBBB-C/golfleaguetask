# Golf League Task

A draft and lineup plan for a team in a women's alternate-shot golf league, built from shot-level data. The original brief is in [TASK.md](TASK.md).

## How I thought about it

The final goal is to draft ten players. So we can start from that decision and work backwards.


To draft well, we need to know how good each player really is, not her seed. Partners take turns hitting one ball, so we also need to know what (which type of hitting) each player is good at: driving, approach, or putting. And single rounds are noisy, so we need to separate skill from luck.

That means knowing what kind of shot every stroke was. The data does not say. It gives an anonymised lie code and a distance, and no par. So before any modelling, the lies and par have to be worked out , using golf's own rules as constraints: every hole starts on a tee and ends in the cup, putts happen on the green, and a penalty follows trouble.

To ensure the data quality,there are few things need cross validation. As for example, the seed caps might leave good players undervalued in the lower tiers; complementary styles might make stronger pairs; the order of the five pairs might matter, as in Tian Ji's horse race. And volatile players might suit an underdog. The notebooks test each one.

## How the work runs


1. **01** rebuilds each hole, decodes the lie codes, checks data quality, infers par and labels every shot.
2. **02** measures every shot against the field (Strokes Gained) and separates each player's skill from luck and field strength.
3. **03** builds a shot-by-shot simulator of alternate-shot match play and checks it against real scoring.
4. **04** uses the simulator to see what pairing and lineup order are worth.
5. **05** drafts, using the ratings and simulations of how the rest of the draft will go.

In short: match play carries so much luck that pairing and order add little. The value is in picking the right players, and the seed misses many of them.


## Setup

```bash
conda env create -f environment.yml
conda activate goalf
# put the data file at data/shot_data.csv
jupyter notebook
```

Run the notebooks in order from the `notebooks/` folder. Notebook 02 saves a cache in `data/processed/` that the later notebooks use.
