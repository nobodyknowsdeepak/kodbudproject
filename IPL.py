"""
IPL Dataset - Exploratory Data Analysis (Simple Version)
Insights: Most Winning Teams | Top Scorers | Stadium Trends
"""

import pandas as pd

# Show ALL rows and columns — no truncation
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------
# You can change the file name/path here
file_path = "C:/Users/Nonu/OneDrive/Desktop/Kodbud Projects/matches-selected-columns.csv"

df = pd.read_csv(file_path)

# Display ALL data
print("\n===== FULL DATASET =====")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nColumns:", df.columns.tolist())
print("\n", df.to_string(index=True))

# ---------------------------------------------------------
# 2. Insight 1: Most Winning Teams
# ---------------------------------------------------------
team_wins = df["match_winner"].value_counts()
print("\nWins per team:")
print(team_wins)

plt.figure(figsize=(8, 5))
plt.bar(team_wins.index, team_wins.values, color="skyblue")
plt.title("Most Winning Teams")
plt.xlabel("Team")
plt.ylabel("Number of Wins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("1_most_winning_teams.png")
plt.show()

# ---------------------------------------------------------
# 3. Insight 2: Top Scorers
# ---------------------------------------------------------
# players who appear most often as top scorer
top_scorer_count = df["top_scorer"].value_counts().head(10)
print("\nMost frequent top scorers:")
print(top_scorer_count)

plt.figure(figsize=(8, 5))
plt.barh(top_scorer_count.index, top_scorer_count.values, color="orange")
plt.title("Players With Most 'Top Scorer' Appearances")
plt.xlabel("Number of Times")
plt.gca().invert_yaxis()  # highest value on top
plt.tight_layout()
plt.savefig("2_top_scorers.png")
plt.show()

# highest individual scores in the season
df["highscore"] = pd.to_numeric(df["highscore"], errors="coerce")
top_scores = df.sort_values("highscore", ascending=False).head(10)

plt.figure(figsize=(8, 5))
plt.barh(top_scores["top_scorer"], top_scores["highscore"], color="green")
plt.title("Highest Individual Scores")
plt.xlabel("Runs")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("3_highest_scores.png")
plt.show()

# ---------------------------------------------------------
# 4. Insight 3: Stadium Trends
# ---------------------------------------------------------
# get stadium name only (before the comma)
df["stadium"] = df["venue"].str.split(",").str[0]

matches_per_stadium = df["stadium"].value_counts()
print("\nMatches hosted per stadium:")
print(matches_per_stadium)

plt.figure(figsize=(8, 5))
plt.bar(matches_per_stadium.index, matches_per_stadium.values, color="purple")
plt.title("Matches Hosted per Stadium")
plt.xlabel("Stadium")
plt.ylabel("Number of Matches")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("4_matches_per_stadium.png")
plt.show()

# average first innings score per stadium
df["first_ings_score"] = pd.to_numeric(df["first_ings_score"], errors="coerce")
avg_score_per_stadium = df.groupby("stadium")["first_ings_score"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(avg_score_per_stadium.index, avg_score_per_stadium.values, color="teal")
plt.title("Average 1st Innings Score per Stadium")
plt.xlabel("Stadium")
plt.ylabel("Average Runs")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("5_avg_score_per_stadium.png")
plt.show()

print("\nDone! All charts have been saved as PNG files.")