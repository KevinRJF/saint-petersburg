import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

// Imports for the modern Smile library (v2.0+)
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructType;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.data.vector.ValueVector;
import smile.regression.RandomForest;

public class OKBasedFeatures {

    String modelFilename = "OKBasedModel.model";
    RandomForest model;
    ArrayList<SPFeature> features;
    StructType schema;

    public OKBasedFeatures() {
        features = new ArrayList<>();
        
        // --- Complete Feature List ---
        features.add(new MyFeatureMinDeckSize());
        features.add(new MyFeaturePoints());
        features.add(new SPFeatureInteractionTerm(new MyFeaturePoints(), new MyFeatureMinDeckSize()));
        features.add(new MyFeaturePointsDiff());
        features.add(new SPFeatureInteractionTerm(new MyFeaturePointsDiff(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureRubles());
        features.add(new SPFeatureInteractionTerm(new MyFeatureRubles(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureRublesDiff());
        features.add(new SPFeatureInteractionTerm(new MyFeatureRublesDiff(), new MyFeatureMinDeckSize()));
        features.add(new MyFeaturePointsRoundGain());
        features.add(new SPFeatureInteractionTerm(new MyFeaturePointsRoundGain(), new MyFeatureMinDeckSize()));
        features.add(new MyFeaturePointsRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new MyFeaturePointsRoundGainDiff(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureRublesRoundGain());
        features.add(new SPFeatureInteractionTerm(new MyFeatureRublesRoundGain(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureRublesRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new MyFeatureRublesRoundGainDiff(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureUniqueAristocrats());
        features.add(new SPFeatureInteractionTerm(new MyFeatureUniqueAristocrats(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureUniqueAristocratsDiff());
        features.add(new SPFeatureInteractionTerm(new MyFeatureUniqueAristocratsDiff(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureCardsInHand());
        features.add(new SPFeatureInteractionTerm(new MyFeatureCardsInHand(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureCardsInHandDiff());
        features.add(new SPFeatureInteractionTerm(new MyFeatureCardsInHandDiff(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureSpotsLeft());
        features.add(new SPFeatureInteractionTerm(new MyFeatureSpotsLeft(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureStartsNext());
        features.add(new SPFeatureInteractionTerm(new MyFeatureStartsNext(), new MyFeatureMinDeckSize()));
        
        // --- Added Features from OKStateFeaturesLR1 ---
        features.add(new MyFeatureAristocratBonus());
        features.add(new SPFeatureInteractionTerm(new MyFeatureAristocratBonus(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureFutureRubleGain());
        features.add(new SPFeatureInteractionTerm(new MyFeatureFutureRubleGain(), new MyFeatureMinDeckSize()));
        features.add(new MyFeatureROI());
        features.add(new SPFeatureInteractionTerm(new MyFeatureROI(), new MyFeatureMinDeckSize()));
        
        initializeModel();
    }
    
    /**
     * Loads the model from the file, or trains a new one if it doesn't exist.
     */
    private void initializeModel() {
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file '" + modelFilename + "' does not exist. Generating a new model...");
            learnModel();
        }
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(new java.io.FileInputStream(modelFilename))) {
            model = (RandomForest) ois.readObject();
            schema = (StructType) ois.readObject();
            System.out.println("Successfully loaded model from '" + modelFilename + "'.");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * Generates training data by simulating games and writing the feature values to a CSV file.
     * Trains on data from SPPlayerFlatMC for a more robust model.
     */
    public void generateCSVData(String filename, int numGames) {
        System.out.println("Generating training data from " + numGames + " simulated games...");
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println(getCSVHeader());
            for (int i = 0; i < numGames; i++) {
                // Train on data from the professor's AI for higher quality data
                SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPPlayerFlatMC(), new SPPlayerFlatMC());
                writer.print(getCSVRows(transcript));
                if ((i + 1) % 1000 == 0) {
                    System.out.println("...simulated " + (i + 1) + " games.");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Finished generating training data.");
    }

    /**
     * Trains a new RandomForest model on the generated data and saves it to a file.
     */
    public void learnModel() {
        String trainingDataFile = "SPTrainingData.csv";
        int numGames = 10000;
        generateCSVData(trainingDataFile, numGames);

        System.out.println("Reading data and training RandomForest model...");
        try {
            List<String> headers = null;
            List<double[]> values = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            
            try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
                String line = br.readLine(); 
                if (line != null) headers = Arrays.asList(line.split(","));
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split(",");
                    double[] row = new double[parts.length - 1];
                    for (int i = 0; i < row.length; i++) row[i] = Double.parseDouble(parts[i]);
                    values.add(row);
                    labels.add(Integer.parseInt(parts[parts.length - 1]));
                }
            }

            int p = features.size();
            double[][] X = values.toArray(new double[0][]);
            int[] y = labels.stream().mapToInt(i -> i).toArray();
            
            List<ValueVector> vectors = new ArrayList<>();
            for (int j = 0; j < p; j++) {
                double[] col = getColumn(X, j);
                vectors.add(DoubleVector.of(headers.get(j), col));
            }
            vectors.add(IntVector.of(headers.get(p), y));
            
            DataFrame df = DataFrame.of(vectors.toArray(new ValueVector[0]));
            this.schema = df.schema();
            
            Formula formula = Formula.lhs(headers.get(p));
            int nTrees = 200, maxDepth = 10, mtry = (int) Math.sqrt(p);
            model = RandomForest.fit(formula, df, nTrees, mtry, maxDepth);

            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilename))) {
                oos.writeObject(model);
                oos.writeObject(this.schema);
            }
            System.out.println("Successfully trained and saved new model to '" + modelFilename + "'.");
            java.nio.file.Files.delete(java.nio.file.Paths.get(trainingDataFile));
        } catch (Exception e) { e.printStackTrace(); }
    }
    
    /**
     * Uses the trained RandomForest model to predict the win probability for a given state.
     */
    public double predict(SPState state) {
        if (schema == null) {
             throw new IllegalStateException("Model schema is not initialized. Make sure the model has been trained or loaded.");
        }
        double[] featureValues = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object value = features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }
        Tuple tuple = Tuple.of(featureValues, schema.fields());
        return model.predict(tuple);
    }
    
    // --- Helper Methods for Data Handling ---
    public String getCSVHeader() {
        StringBuilder header = new StringBuilder();
        for (SPFeature feature : features) {
            header.append(feature.getName()).append(",");
        }
        header.append("is_winner");
        return header.toString();
    }

    public String getCSVRows(SPGameTranscript transcript) {
        StringBuilder rows = new StringBuilder();
        boolean[] isWinner = transcript.getWinners();
        List<SPState> states = transcript.getStates();
        for(SPState state : states){
             rows.append(getCSVRow(state, isWinner)).append("\n");
        }
        return rows.toString();
    }

    public String getCSVRow(SPState state, boolean[] isWinner) {
        int player = state.playerTurn;
        if (state.isGameOver()) {
            player = (state.playerTurn + state.numPlayers - 1) % state.numPlayers; // Last player to move
        }
        int winnerVal = isWinner[player] ? 1 : 0;
        StringBuilder row = new StringBuilder();
        for (SPFeature feature : features) {
            row.append(feature.getValue(state)).append(",");
        }
        row.append(winnerVal);
        return row.toString();
    }
    
    private static double[] getColumn(double[][] matrix, int columnIndex) {
       double[] column = new double[matrix.length];
       for(int i = 0; i < matrix.length; i++){
          column[i] = matrix[i][columnIndex];
       }
       return column;
    }
    
    // --- Main method to easily trigger the training process ---
    public static void main(String[] args) {
        // Running this file will automatically create and train the model if it doesn't exist.
        new MyRFStateFeatures();
    }

    // ===================================================================================
    //                          FEATURE DEFINITIONS START HERE
    // ===================================================================================

    class MyFeatureMinDeckSize extends SPFeature {
        public MyFeatureMinDeckSize() { super("min_deck_size", "the number of cards in the smallest phase deck"); }
        public Object getValue(SPState state) {
            return Math.min(Math.min(state.workerDeck.size(), state.buildingDeck.size()), Math.min(state.aristocratDeck.size(), state.tradingDeck.size()));
        }
    }
    class MyFeaturePoints extends SPFeature {
        public MyFeaturePoints() { super("points", "current player points"); }
        public Object getValue(SPState state) { return state.playerPoints[state.playerTurn]; }
    }
    class MyFeaturePointsDiff extends SPFeature {
        public MyFeaturePointsDiff() { super("points_diff", "current player points relative to the opponent"); }
        public Object getValue(SPState state) { return state.playerPoints[state.playerTurn] - state.playerPoints[1 - state.playerTurn]; }
    }
    class MyFeatureRubles extends SPFeature {
        public MyFeatureRubles() { super("rubles", "current player rubles (money)"); }
        public Object getValue(SPState state) { return state.playerRubles[state.playerTurn]; }
    }
    class MyFeatureRublesDiff extends SPFeature {
        public MyFeatureRublesDiff() { super("rubles_diff", "current player rubles (money) relative to the opponent"); }
        public Object getValue(SPState state) { return state.playerRubles[state.playerTurn] - state.playerRubles[1 - state.playerTurn]; }
    }
    class MyFeaturePointsRoundGain extends SPFeature {
        public MyFeaturePointsRoundGain() { super("points_round_gain", "the number of points the current player is gaining per round"); }
        public Object getValue(SPState state) {
            return state.playerWorkers.get(state.playerTurn).stream().mapToInt(c -> c.points).sum()
                 + state.playerBuildings.get(state.playerTurn).stream().mapToInt(c -> c.points).sum()
                 + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(c -> c.points).sum();
        }
    }
    class MyFeaturePointsRoundGainDiff extends SPFeature {
        public MyFeaturePointsRoundGainDiff() { super("points_round_gain_diff", "the number of points the current player is gaining per round relative to the opponent"); }
        public Object getValue(SPState state) {
            int p0_gain = state.playerWorkers.get(0).stream().mapToInt(c -> c.points).sum() + state.playerBuildings.get(0).stream().mapToInt(c -> c.points).sum() + state.playerAristocrats.get(0).stream().mapToInt(c -> c.points).sum();
            int p1_gain = state.playerWorkers.get(1).stream().mapToInt(c -> c.points).sum() + state.playerBuildings.get(1).stream().mapToInt(c -> c.points).sum() + state.playerAristocrats.get(1).stream().mapToInt(c -> c.points).sum();
            return state.playerTurn == 0 ? p0_gain - p1_gain : p1_gain - p0_gain;
        }
    }
    class MyFeatureRublesRoundGain extends SPFeature {
        public MyFeatureRublesRoundGain() { super("rubles_round_gain", "the number of rubles the current player is gaining per round"); }
        public Object getValue(SPState state) {
            return state.playerWorkers.get(state.playerTurn).stream().mapToInt(c -> c.rubles).sum()
                 + state.playerBuildings.get(state.playerTurn).stream().mapToInt(c -> c.rubles).sum()
                 + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(c -> c.rubles).sum();
        }
    }
    class MyFeatureRublesRoundGainDiff extends SPFeature {
        public MyFeatureRublesRoundGainDiff() { super("rubles_round_gain_diff", "the number of rubles the current player is gaining per round relative to the opponent"); }
        public Object getValue(SPState state) {
            int p0_gain = state.playerWorkers.get(0).stream().mapToInt(c -> c.rubles).sum() + state.playerBuildings.get(0).stream().mapToInt(c -> c.rubles).sum() + state.playerAristocrats.get(0).stream().mapToInt(c -> c.rubles).sum();
            int p1_gain = state.playerWorkers.get(1).stream().mapToInt(c -> c.rubles).sum() + state.playerBuildings.get(1).stream().mapToInt(c -> c.rubles).sum() + state.playerAristocrats.get(1).stream().mapToInt(c -> c.rubles).sum();
            return state.playerTurn == 0 ? p0_gain - p1_gain : p1_gain - p0_gain;
        }
    }
    class MyFeatureUniqueAristocrats extends SPFeature {
        public MyFeatureUniqueAristocrats() { super("unique_aristocrats", "the number of unique aristocrats of the current player"); }
        public Object getValue(SPState state) { return state.getNumUniqueAristocrats(state.playerTurn); }
    }
    class MyFeatureUniqueAristocratsDiff extends SPFeature {
        public MyFeatureUniqueAristocratsDiff() { super("unique_aristocrats_diff", "the number of unique aristocrats of the current player relative to the opponent"); }
        public Object getValue(SPState state) { return state.getNumUniqueAristocrats(state.playerTurn) - state.getNumUniqueAristocrats(1 - state.playerTurn); }
    }
    class MyFeatureCardsInHand extends SPFeature {
        public MyFeatureCardsInHand() { super("cards_in_hand", "the number of cards in the current player hand"); }
        public Object getValue(SPState state) { return state.playerHands.get(state.playerTurn).size(); }
    }
    class MyFeatureCardsInHandDiff extends SPFeature {
        public MyFeatureCardsInHandDiff() { super("cards_in_hand_diff", "the number of cards in the current player hand relative to the opponent"); }
        public Object getValue(SPState state) { return state.playerHands.get(state.playerTurn).size() - state.playerHands.get(1 - state.playerTurn).size(); }
    }
    class MyFeatureSpotsLeft extends SPFeature {
        public MyFeatureSpotsLeft() { super("spots_left", "how many spots there are for the cards to be shown in the next round"); }
        public Object getValue(SPState state) { return 8 - state.upperCardRow.size() - state.lowerCardRow.size(); }
    }
    class MyFeatureStartsNext extends SPFeature {
        public MyFeatureStartsNext() { super("starts_next", "whether the player starts the next phase"); }
        public Object getValue(SPState state) {
            if (state.phase >= SPState.TRADING) return 0; // End of round or special phase
            int nextStartingPlayer = state.startingPlayer[state.phase + 1];
            return nextStartingPlayer == state.playerTurn ? 1 : 0;
        }
    }

    /**
     * Calculates the actual end-game bonus points a player would receive.
     * FIX: Includes a bounds check to prevent crashes.
     */
    class MyFeatureAristocratBonus extends SPFeature {
        public MyFeatureAristocratBonus() {
            super("aristocrat_bonus", "Calculates the end-game bonus points for unique aristocrats");
        }
        @Override
        public Object getValue(SPState state) {
            int numUniqueAristocrats = state.getNumUniqueAristocrats(state.playerTurn);
            int maxIndex = SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.size() - 1;
            int indexToUse = Math.min(numUniqueAristocrats, maxIndex);
            return SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get(indexToUse);
        }
    }

    /**
     * Calculates the total potential income (rubles-per-round) from all
     * affordable cards currently available on the board.
     */
    class MyFeatureFutureRubleGain extends SPFeature {
        public MyFeatureFutureRubleGain() {
            super("future_ruble_gain", "Sum of rubles-per-round from affordable marketplace cards");
        }
        @Override
        public Object getValue(SPState state) {
            int playerRubles = state.playerRubles[state.playerTurn];
            int futureGain = 0;
            for (SPCard card : state.upperCardRow) {
                if (card.cost <= playerRubles) {
                    futureGain += card.rubles;
                }
            }
            for (SPCard card : state.lowerCardRow) {
                if ((card.cost - 1) <= playerRubles) {
                    futureGain += card.rubles;
                }
            }
            return futureGain;
        }
    }

    /**
     * A sophisticated ROI feature that estimates the total points a player might
     * end up with, divided by an estimate of the number of rounds left in the game.
     */
    class MyFeatureROI extends SPFeature {
        private final int samples = 50;
        private final int maxDepth = 40;

        public MyFeatureROI() {
            super("estimated_points_per_round", "Estimated final points divided by estimated rounds remaining");
        }
        @Override
        public Object getValue(SPState state) {
            double expectedPoints = estimatePotentialPoints(state);
            double estRoundsLeft = estimateRoundsRemaining(state);
            return expectedPoints / Math.max(1.0, estRoundsLeft);
        }
        private double estimateRoundsRemaining(SPState root) {
            int totalTurns = 0;
            for (int s = 0; s < samples; s++) {
                SPState sim = root.clone();
                int turns = 0;
                while (!sim.isGameOver() && turns < maxDepth) {
                    ArrayList<SPAction> legal = sim.getLegalActions();
                    if (legal.isEmpty()) break;
                    legal.get((int) (Math.random() * legal.size())).take();
                    turns++;
                }
                totalTurns += turns;
            }
            return ((double) totalTurns / (double) samples) / root.numPlayers;
        }
        private double estimatePotentialPoints(SPState state) {
            int player = state.playerTurn;
            int currentPoints = state.playerPoints[player];
            int numUniqueAristocrats = state.getNumUniqueAristocrats(player);
            int maxBonusIndex = SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.size() - 1;
            int aristocratBonus = SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get(Math.min(numUniqueAristocrats, maxBonusIndex));
            return (double) currentPoints + aristocratBonus;
        }
    }
}