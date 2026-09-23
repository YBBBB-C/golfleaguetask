# Candidate Technical Brief: Women’s Golf League Draft & Simulation

## Context & Competition Mechanics

You are acting as the Lead Data Scientist for a team in a hypothetical women’s professional team golf league. This document outlines all league formats, draft rules, golf fundamentals, and scoring mechanics from first principles.

### Fundamentals of Golf & Shot Categories

Golf is played on a course of 18 distinct holes. The goal on each hole is to propel the ball from the starting tee into the hole in as few strokes (shots) as possible.

* **Par:** Every hole is assigned a "Par" rating—the benchmark number of strokes an expert golfer is expected to need to complete the hole (typically Par 3, Par 4, or Par 5).
* **Course Terrain & Lies:**
	* **Tee Box:** The starting surface for the first shot of each hole.
	* **Fairway:** Maintained, short-cut grass optimal for advancing shots toward the green.
	* **Rough:** Higher, unkept grass lining the fairway that makes contact and control difficult.
	* **Sand Bunker:** Hazard pits filled with sand requiring specialized recovery shots.
	* **Green:** Extremely smooth, short-cut grass surrounding the hole designed for putting.
* **Game Areas / Shot Types:**
	* **Off-the-Tee (Driving):** The long-distance initial shot from the tee box on Par 4 and Par 5 holes.
	* **Approach:** Shots taken from distance (>100 yards) aimed at landing the ball on the green.
	* **Around-the-Green (Short Game):** Touch shots, chips, pitches, and bunker saves taken from within ~100 yards of the green.
	* **Putting:** Rolling shots struck on the green surface into the cup.

### Roster & Fixture Structure

* **Rosters:** Each team maintains a roster of 10 players.
* **Fixtures:** A fixture is a head-to-head match between two teams consisting of 5 distinct sub-matches played between 2-player pairs.
* **Sub-Match Length:** Every sub-match is scheduled for **18 holes**.
* **Alternate-Shot Format:** In each sub-match, two teammates share a single ball and alternate taking shots until the ball is holed (e.g., Player A takes shot 1, Player B takes shot 2, Player A takes shot 3, and so on).
* **Tee Shot Alternation Rule:** Players must strictly alternate tee shots by hole. One teammate tees off on all odd-numbered holes (1, 3, 5, etc.) while the other tees off on all even-numbered holes (2, 4, 6, etc.). **This rule applies strictly regardless of which player hit the final shot/putt on the previous hole.**

### Match Play Scoring Mechanics

Sub-matches are played using hole-by-hole "match play" scoring rather than total cumulative strokes across the 18 holes:

* **Hole Outcomes:** The pair that completes a hole in fewer alternate shots wins that hole, moving their sub-match score to **1 Up**. If both pairs take the same number of shots, the hole is **halved** (tied), and the running score remains unchanged.
* **Running Score:** Scores reflect the net difference in holes won (e.g., **2 Up** means Pair A has won two more holes than Pair B; **All Square** means the sub-match is tied).
* **Sub-Match Early Termination:** Sub-matches are scheduled for 18 holes but terminate immediately as soon as one pair leads by more holes than there are remaining to play. For example, if Pair A is **3 Up** with only 2 holes left, Pair A wins **3 & 2**, and the remaining holes (17 and 18) are not played. If Pair A leads by 1 hole after all 18 holes, they win **1 Up**. If tied after 18 holes, the sub-match is a draw (**Halved**).
* **Fixture Points & Early Closure:**
	* Sub-match Win = 1.0 fixture point
	* Sub-match Draw = 0.5 fixture points to each team
	* Sub-match Loss = 0.0 fixture points
The overall team fixture ends immediately as soon as a team reaches **3.0 fixture points** (e.g., winning three sub-matches 3–0, or securing two wins and two draws).

### Step-by-Step Match Walkthrough (Holes 1–3)

To illustrate how alternate-shot execution and match play scoring work in practice over an 18-hole sub-match, consider a contest between **Pair A** (Player A1 & Player A2) and **Pair B** (Player B1 & Player B2).

*Tee shot assignment: Player A1 and Player B1 tee off on odd holes (1, 3, 5...); Player A2 and Player B2 tee off on even holes (2, 4, 6...).*

* **Hole 1 (Par 4 — Odd Hole)**
	* *Shot 1 (Tee Shot):* Player A1 drives onto the fairway. Player B1 drives into the rough.
	* *Shot 2 (Approach):* Player A2 hits onto the green. Player B2 hits out of the rough into a sand bunker.
	* *Shot 3:* Player A1 putts close to the hole. Player B2 chips out of the bunker onto the green.
	* *Shot 4:* **Player A2 taps in for a total of 4 strokes.** Player B1 putts and misses.
	* *Shot 5:* Player B2 taps in for a total of **5 strokes**.
* **Hole 1 Result:** Pair A wins the hole (4 strokes vs 5 strokes).
* **Running Score:** **Pair A is 1 Up**.

* **Hole 2 (Par 3 — Even Hole)**
	* *Shot 1 (Tee Shot):* **Player A2 must tee off** because Hole 2 is an even hole—even though Player A2 hit the final tap-in shot on Hole 1. Player A2 hits the green. Player B2 hits the green.
	* *Shot 2 (Putting):* Player A1 putts past the hole. Player B1 putts close.
	* *Shot 3 (Tapping in):* Player A2 taps in for a total of **3 strokes**. Player B2 taps in for a total of **3 strokes**.
* **Hole 2 Result:** The hole is **Halved** (tied at 3 strokes each).
* **Running Score:** **Pair A remains 1 Up**.

* **Hole 3 (Par 5 — Odd Hole)**
	* *Shot 1 (Tee Shot):* Tee shots reset to Player A1 and Player B1 for the odd hole. Player A1 drives onto the fairway; Player B1 drives far down the fairway.
	* *Shot 2:* Player A2 lays up safely. Player B2 reaches the green in two shots.
	* *Shot 3:* Player A1 hits onto the green. Player B1 putts close.
	* *Shot 4:* Player A2 putts and misses. Player B2 taps in for **4 strokes**.
* **Hole 3 Result:** Pair B wins the hole (4 strokes vs 5 strokes).
* **Running Score:** **All Square** (tied match).

### Tactical Lineup Deployment

Before a fixture begins, teams simultaneously submit their 5 pairs in a specific order (Pair 1 through Pair 5). Pair 1 plays against the opponent's Pair 1, Pair 2 against Pair 2, and so on.

### League Reset & Snake Draft Mechanics

At the start of the season, all $N$ teams in the league start with empty rosters and build their 10-player squads entirely from scratch out of a unified pool of unpicked players.

* **Draft Pool:** All available players are initially unassigned. No team retains legacy players.
* **Snake Draft Format:** To ensure fairness across draft positions, selection order reverses at the end of every round ("snaking" back and forth):
	* Round 1 (Odd Rounds: 1, 3, 5, 7, 9): Sequential ascending pick order (Team 1, Team 2, ..., Team $N$)
	* Round 2 (Even Rounds: 2, 4, 6, 8, 10): Reversed descending pick order (Team $N$, Team $N-1$, ..., Team 1)
	* Round 3: Ascending pick order repeats (Team 1, Team 2, ..., Team $N$)
* This back-and-forth order continues across 10 rounds until every team has selected exactly 10 players.

* **Draft Constraints (Performance Caps):** As picks unfold, every team's cumulative roster must respect bucket caps based on last season's official player rankings:
	* Top 10 Tier (Rank 1–10): Maximum 2 players
	- Top 20 Tier (Rank 1–20): Maximum 4 players total (cumulative)
	- Top 30 Tier (Rank 1–30): Maximum 6 players total (cumulative)
	- Top 40 Tier (Rank 1–40): Maximum 8 players total (cumulative)
	- Top 50 Tier (Rank 1–50): Maximum 10 players total (cumulative)
	- Top 60 to Top 100 Tiers (Rank 51–100+): Unrestricted (subject to the overall 10-player roster limit)
	  
## Provided Datasets

All shot-level execution metrics, tournament metadata, and official baseline player rankings are provided in a single consolidated file: `shot_data.csv`. 

Columns: 
- **`player_id` & `draft_seed`:** Anonymised player identifier and official baseline draft seed (1 to 105) used to enforce draft bucket caps.
- **Event & Hierarchy Metadata:**
    - `event_id`, `event_start_date`, `event_end_date`, `year`: Tournament dates and identifiers spanning multiple seasons (2022–2023).
    - `course_no`, `round_no`, `hole_no`, `shot_no`, `shot_index`: Sequential tracking variables defining round, hole, and shot order.
- **Shot Execution & Lie Categories:**
    - `lie_before` & `lie_after`: Categorised surface/lie conditions prior to and following the stroke (`LIE_1` through `LIE_19`).
    - `club`: Equipment recorded for the stroke (where available).
    - `drop_type` & `penalty_type`: Recorded rules violations and drop flags.
- **Spatial Distance Metrics (in Inches):**
    - `distance_before_inch`: Initial distance remaining from the ball to the target prior to the shot.
    - `distance_after_inch`: Distance remaining to the target after shot completion.
    - `distance`: Total distance covered by the shot.

## Candidate Approach & Scope

You have complete autonomy over how you structure this project, how deep you go into any specific mechanic, and how you choose to present your work.

To provide inspiration, here are several directions you might choose to explore, combine, or build upon:

* **Player Performance Modelling:** Quantifying underlying player quality across distinct game areas (e.g., long-range driving, approach shots, short game, putting) using strokes gained logic or custom shot distributions directly from shot-level data.
* **Stylistic Alternate-Shot Pairing:** Designing an algorithmic framework to construct complementary 2-player pairings (e.g., pairing a long-distance driver with a high-accuracy approach player, or optimising who takes odd vs. even tee shots based on course layout).
* **Draft Optimisation & Contingencies:** Developing a draft decision engine that optimises roster strength within performance caps and dynamically adapts target lists as opponent picks unfold in the snake draft.
* **Fixture Simulation & Tactical Lineups:** Building a Monte Carlo 18-hole match play simulator to evaluate sub-match outcomes, test lineup ordering strategies, and calculate team win probabilities against prospective opponent lineups.

If you choose to go into a lot of detail on just one or two of these aspects, please be prepared to discuss during the interview how your chosen focus fits into the wider overall problem scope and end-to-end system design.

## Deliverables & Interview Walkthrough

Please submit your codebase or notebook prior to the technical interview stage.

During our follow-up session, we will treat your submission as a collaborative springboard for discussion. We will walk through your code together, exploring your mathematical choices, architectural decisions, handling of edge cases, and the underlying rationale behind your implementation.