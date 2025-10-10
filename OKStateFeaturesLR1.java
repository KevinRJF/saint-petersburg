
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import smile.classification.LogisticRegression;

/**
 * OKStateFeaturesLR1 - feature extractor + logistic-win model.
 * Includes ROI Engineered feature
 */
public class OKStateFeaturesLR1 {
    String modelFilename = "OKLogisticRegression1.model";
    LogisticRegression.Binomial model;
    ArrayList<SPFeature> features;

    public ArrayList<Object> getFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : features) {
            values.add(feature.getValue(state));
        }
        return values;
    }

    public OKStateFeaturesLR1() {
        features = new ArrayList<>();
        features.add(new SPFeatureMinDeckSize());
        features.add(new SPFeaturePoints());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePoints(), new SPFeatureMinDeckSize()));
        features.add(new SPFeaturePointsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRubles());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRubles(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRublesDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeaturePointsRoundGain());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGain(), new SPFeatureMinDeckSize()));
        features.add(new SPFeaturePointsRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGainDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRublesRoundGain());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGain(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureRublesRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGainDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureUniqueAristocrats());
        features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocrats(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureUniqueAristocratsDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocratsDiff(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureCardsInHand());
        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHand(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureCardsInHandDiff());
        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHandDiff(), new SPFeatureMinDeckSize()));
        // ROI feature (fixed implementation)
        features.add(new ROIFeature()); // ~index 25
        features.add(new SPFeatureInteractionTerm(new ROIFeature(), new SPFeatureMinDeckSize()));
        initializeModel();
    }

    private void initializeModel() {
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file does not exist. Generating model...");
            learnModel();
        }
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(
                new java.io.FileInputStream(modelFilename))) {
            model = (LogisticRegression.Binomial) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public String getCSVHeader() {
        StringBuilder header = new StringBuilder();
        for (SPFeature feature : features) {
            header.append(feature.getName()).append(",");
        }
        header.append("is_winner");
        return header.toString();
    }

    public String getCSVRow(SPState state, boolean[] isWinner) {
        int currentPlayerIndex = state.playerTurn;
        int winnerVal = isWinner[currentPlayerIndex] ? 1 : 0;
        StringBuilder row = new StringBuilder();
        for (SPFeature feature : features) {
            row.append(feature.getValue(state)).append(",");
        }
        row.append(winnerVal);
        return row.toString();
    }

    public String getCSVRows(SPGameTranscript transcript) {
        StringBuilder rows = new StringBuilder();
        boolean[] isWinner = transcript.getWinners();
        for (SPState state : transcript.getStates()) {
            rows.append(getCSVRow(state, isWinner)).append("\n");
        }
        return rows.toString();
    }

    public void generateCSVData(String filename, int numGames) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println(getCSVHeader());
            for (int i = 0; i < numGames; i++) {
                SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPPlayerFlatMC(), new SPPlayerFlatMC());
                writer.print(getCSVRows(transcript));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void learnModel() {
        // WARNING: expensive. Lower numGames while debugging.
        String trainingDataFile = "SPTrainingData.csv";
        int numGames = 1000; // adjust down for development
        generateCSVData(trainingDataFile, numGames);

        List<double[]> values = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
            String line = br.readLine(); // header
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] row = new double[parts.length - 1];
                for (int i = 0; i < row.length; i++) {
                    row[i] = Double.parseDouble(parts[i]);
                }
                values.add(row);
                labels.add(Integer.parseInt(parts[parts.length - 1]));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[][] X = values.toArray(new double[0][]);
        int[] y = labels.stream().mapToInt(i -> i).toArray();

        // Train logistic regression (win-probability)
        LogisticRegression.Binomial modelLocal = LogisticRegression.binomial(X, y);

        // Save model
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(
                new java.io.FileOutputStream(modelFilename))) {
            oos.writeObject(modelLocal);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Model coefficients:");
        System.out.println(features.size() + " features");
        System.out.println(modelLocal.coefficients().length + " coefficients");
        System.out.printf("%.4f\tIntercept%n", modelLocal.coefficients()[0]);
        for (int i = 0; i < modelLocal.coefficients().length - 1; i++) {
            System.out.printf("%.4f\t%s%n", modelLocal.coefficients()[i + 1], features.get(i).getName());
        }

        // delete training data file
        java.nio.file.Path path = java.nio.file.Paths.get(trainingDataFile);
        try {
            java.nio.file.Files.delete(path);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double predict(SPState state) {
        double[] featureValues = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object value = features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }
        return model.score(featureValues);
    }

    /* --------------------- Inner feature classes --------------------- */

    // Interaction term: product of two features
    class SPFeatureInteractionTerm extends SPFeature {
        private final SPFeature a;
        private final SPFeature b;

        public SPFeatureInteractionTerm(SPFeature a, SPFeature b) {
            super(a.getName() + "_x_" + b.getName(), "interaction of " + a.getName() + " and " + b.getName());
            this.a = a;
            this.b = b;
        }

        @Override
        public Object getValue(SPState state) {
            Object va = a.getValue(state);
            Object vb = b.getValue(state);
            double da = (va instanceof Number) ? ((Number) va).doubleValue() : 0.0;
            double db = (vb instanceof Number) ? ((Number) vb).doubleValue() : 0.0;
            return da * db;
        }
    }

    // min_deck_size
    class SPFeatureMinDeckSize extends SPFeature {
        public SPFeatureMinDeckSize() {
            super("min_deck_size", "the number of cards in the smallest phase deck");
        }

        public Object getValue(SPState state) {
            int minDeckSize = Integer.MAX_VALUE;
            minDeckSize = Math.min(minDeckSize, state.workerDeck.size());
            minDeckSize = Math.min(minDeckSize, state.buildingDeck.size());
            minDeckSize = Math.min(minDeckSize, state.aristocratDeck.size());
            minDeckSize = Math.min(minDeckSize, state.tradingDeck.size());
            return minDeckSize;
        }
    }

    // points
    class SPFeaturePoints extends SPFeature {
        public SPFeaturePoints() {
            super("points", "current player points");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn];
        }
    }

    // points_diff
    class SPFeaturePointsDiff extends SPFeature {
        public SPFeaturePointsDiff() {
            super("points_diff", "current player points relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn] - state.playerPoints[1 - state.playerTurn];
        }
    }

    // rubles
    class SPFeatureRubles extends SPFeature {
        public SPFeatureRubles() {
            super("rubles", "current player rubles (money)");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn];
        }
    }

    // rubles_diff
    class SPFeatureRublesDiff extends SPFeature {
        public SPFeatureRublesDiff() {
            super("rubles_diff", "current player rubles (money) relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn] - state.playerRubles[1 - state.playerTurn];
        }
    }

    // points_round_gain
    class SPFeaturePointsRoundGain extends SPFeature {
        public SPFeaturePointsRoundGain() {
            super("points_round_gain", "the number of points the current player is gaining per round");
        }

        public Object getValue(SPState state) {
            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
            return pointsPerRound;
        }
    }

    // points_round_gain_diff
    class SPFeaturePointsRoundGainDiff extends SPFeature {
        public SPFeaturePointsRoundGainDiff() {
            super("points_round_gain_diff",
                    "the number of points the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
            int opponentPointsPerRound = state.playerWorkers.get(1 - state.playerTurn).stream()
                    .mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum();
            return pointsPerRound - opponentPointsPerRound;
        }
    }

    // rubles_round_gain
    class SPFeatureRublesRoundGain extends SPFeature {
        public SPFeatureRublesRoundGain() {
            super("rubles_round_gain", "the number of rubles the current player is gaining per round");
        }

        public Object getValue(SPState state) {
            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            return rublesPerRound;
        }
    }

    // rubles_round_gain_diff
    class SPFeatureRublesRoundGainDiff extends SPFeature {
        public SPFeatureRublesRoundGainDiff() {
            super("rubles_round_gain_diff",
                    "the number of rubles the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            int opponentRublesPerRound = state.playerWorkers.get(1 - state.playerTurn).stream()
                    .mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            return rublesPerRound - opponentRublesPerRound;
        }
    }

    // unique_aristocrats
    class SPFeatureUniqueAristocrats extends SPFeature {
        public SPFeatureUniqueAristocrats() {
            super("unique_aristocrats", "the number of unique aristocrats of the current player");
        }

        public Object getValue(SPState state) {
            return state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
        }
    }

    // unique_aristocrats_diff
    class SPFeatureUniqueAristocratsDiff extends SPFeature {
        public SPFeatureUniqueAristocratsDiff() {
            super("unique_aristocrats_diff",
                    "the number of unique aristocrats of the current player relative to the opponent");
        }

        public Object getValue(SPState state) {
            long uniqueAristocrats = state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
            long opponentUniqueAristocrats = state.playerAristocrats.get(1 - state.playerTurn).stream().distinct()
                    .count();
            return uniqueAristocrats - opponentUniqueAristocrats;
        }
    }

    // cards_in_hand
    class SPFeatureCardsInHand extends SPFeature {
        public SPFeatureCardsInHand() {
            super("cards_in_hand", "the number of cards in the current player hand");
        }

        public Object getValue(SPState state) {
            return state.playerHands.get(state.playerTurn).size();
        }
    }

    // cards_in_hand_diff
    class SPFeatureCardsInHandDiff extends SPFeature {
        public SPFeatureCardsInHandDiff() {
            super("cards_in_hand_diff", "the number of cards in the current player hand relative to the opponent");
        }

        public Object getValue(SPState state) {
            int cardsInHand = state.playerHands.get(state.playerTurn).size();
            int opponentCardsInHand = state.playerHands.get(1 - state.playerTurn).size();
            return cardsInHand - opponentCardsInHand;
        }
    }

    /**
     * ROIFeature - corrected and safer.
     * Estimates expected points and divides by estimated remaining turns
     * (averaged).
     */
    class ROIFeature extends SPFeature {
        private final int samples = 50; // playouts for estimating rounds left
        private final int maxDepth = 30; // cap for playout length

        public ROIFeature() {
            super("ROI", "expected points per remaining round with income awareness");
        }

        @Override
        public Object getValue(SPState state) {
            int player = state.playerTurn;
            double expectedPoints = estimatePoints(state, player);
            double expectedIncome = estimateIncome(state, player);
            double estRoundsLeft = estimateRounds(state);

            // Phase-aware weighting
            double phaseFactor = (state.phase <= 3) ? 0.5 : 1.0; // early game: emphasize income

            // ROI = (points + weighted income) / estRoundsLeft
            double roi = (expectedPoints + phaseFactor * expectedIncome) / Math.max(1.0, estRoundsLeft);

            return roi;
        }

        // Estimate remaining rounds using playouts
        private double estimateRounds(SPState root) {
            int totalTurns = 0;
            for (int s = 0; s < samples; s++) {
                SPState sim = root.clone();
                int turns = 0;
                while (!sim.isGameOver() && turns < maxDepth) {
                    ArrayList<SPAction> legal = sim.getLegalActions();
                    if (legal.isEmpty())
                        break;
                    legal.get((int) (Math.random() * legal.size())).take();
                    turns++;
                }
                totalTurns += turns;
            }
            return (double) totalTurns / samples;
        }

        // Estimate points: current points + VP from best 2–3 cards + aristocrats
        private double estimatePoints(SPState state, int player) {
            int currentPoints = state.playerPoints[player];

            // Best 3 VP cards in legal actions
            int bestVP = 0;
            int count = 0;
            ArrayList<SPAction> legal = state.getLegalActions();
            ArrayList<Integer> vpValues = new ArrayList<>();
            for (SPAction action : legal) {
                if (action instanceof SPBuyAction) {
                    SPCard card = ((SPBuyAction) action).card;
                    if (card != null)
                        vpValues.add(card.points);
                }
            }
            vpValues.sort((a, b) -> b - a); // descending
            for (int vp : vpValues) {
                if (count >= 3)
                    break;
                bestVP += vp;
                count++;
            }

            // Aristocrat bonus
            int aristocratBonus = 0;
            Set<String> uniqueAris = new HashSet<>();
            for (SPCard c : state.playerAristocrats.get(player)) {
                if (c != null && c.isAristocrat)
                    uniqueAris.add(c.name);
            }
            int n = uniqueAris.size();
            if (n > 0) {
                if (SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS != null
                        && !SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.isEmpty()) {
                    aristocratBonus = SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS
                            .get(Math.min(n, SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.size() - 1));
                } else {
                    int[] bonusTable = { 0, 1, 3, 6, 10, 15, 21, 28, 36 };
                    aristocratBonus = bonusTable[Math.min(n, bonusTable.length - 1)];
                }
            }

            return currentPoints + bestVP + aristocratBonus;
        }

        // Estimate income: rubles from owned cards + top 3 buyable cards
        private double estimateIncome(SPState state, int player) {
            int income = 0;

            // Income from current cards
            income += state.playerWorkers.get(player).stream().mapToInt(c -> c.rubles).sum();
            income += state.playerBuildings.get(player).stream().mapToInt(c -> c.rubles).sum();
            income += state.playerAristocrats.get(player).stream().mapToInt(c -> c.rubles).sum();

            // Potential income from top 3 affordable cards
            int playerRubles = state.playerRubles[player];
            ArrayList<Integer> potentialIncome = new ArrayList<>();
            for (SPAction action : state.getLegalActions()) {
                if (action instanceof SPBuyAction) {
                    SPCard card = ((SPBuyAction) action).card;
                    if (card != null && card.cost <= playerRubles) {
                        potentialIncome.add(card.rubles);
                    }
                }
            }
            potentialIncome.sort((a, b) -> b - a); // descending
            for (int i = 0; i < Math.min(3, potentialIncome.size()); i++) {
                income += potentialIncome.get(i);
            }

            return income;
        }
    }

    // Inside OKStateFeaturesRF1.java, after other feature classes
    class SPFeatureFutureRubleGain extends SPFeature {
        public SPFeatureFutureRubleGain() {
            super("future_ruble_gain", "sum of rubles-per-round from affordable marketplace cards");
        }

        @Override
        public Object getValue(SPState state) {
            int playerRubles = state.playerRubles[state.playerTurn];
            int futureGain = 0;

            // Check Upper Card Row
            for (SPCard card : state.upperCardRow) {
                if (card.cost <= playerRubles) {
                    futureGain += card.rubles;
                }
            }

            // Check Lower Card Row
            for (SPCard card : state.lowerCardRow) {
                if (card.cost <= playerRubles) {
                    futureGain += card.rubles;
                }
            }

            return futureGain;
        }
    }

    // main for quick smoke test
    public static void main(String[] args) {
        new OKStateFeaturesLR1();
        System.out.println("OKStateFeaturesLR1 constructed.");
    }
}
