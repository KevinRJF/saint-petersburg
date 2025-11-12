import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;


// This class, like SPStateFeaturesLR1.java, seeks to predict the probability of winning.
// However, instead of using logistic regression, it uses a neural network model from the DJ4J library.


public class OKStateFeaturesNN1 {
    String modelFilename = "OKNN2.model";
    ArrayList<SPFeature> features;
    // Serializable DL4J neural network model
    private MultiLayerNetwork model;

    public ArrayList<Object> getFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : features) {
            values.add(feature.getValue(state));
        }
        return values;
    }

    public OKStateFeaturesNN1() {
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
        features.add(new ReserveRublesFeature());
        features.add(new SPFeatureInteractionTerm(new ReserveRublesFeature(), new SPFeatureMinDeckSize()));
        features.add(new DontBeAnIdiotFeature());
        features.add(new SPFeatureInteractionTerm(new DontBeAnIdiotFeature(), new SPFeatureMinDeckSize()));
        features.add(new SPFeatureStrategicBuildingPreference());
        features.add(new SPFeatureInteractionTerm(new SPFeatureStrategicBuildingPreference(), new SPFeatureMinDeckSize()));
        initializeModel();
    }

    private void initializeModel() {
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file does not exist. Generating model...");
            learnModel();
        }

        try {
            model = ModelSerializer.restoreMultiLayerNetwork(new java.io.File(modelFilename));
            System.out.println("Model loaded from " + modelFilename);
            System.out.println(model.summary());
        } catch (IOException e) {
            e.printStackTrace();
        }

        // try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(new java.io.FileInputStream(modelFilename))) {
        //     model = (MultiLayerNetwork) ois.readObject();
        //     System.out.println("Model loaded from " + modelFilename);
        //     System.out.println(model.summary());
        // } catch (IOException | ClassNotFoundException e) {
        //     e.printStackTrace();
        // }
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
                System.out.println(i);
                //System.out.println("okplayerdata");
                SPGameTranscript transcript = SPSimulateGame.simulateGame(new OKTurnBasedFeaturesPlayer(), new OKTurnBasedFeaturesPlayer());
                writer.print(getCSVRows(transcript));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void learnModel() {
        // This method assumes that the logistic regression model has not been created and saved yet.
        // It generates training data by simulating games and saves it to a CSV file.
        // Then it uses logistic regression to learn a model and saves it to a file.

        String trainingDataFile = "SPTrainingDataNN.csv";
        int numGames = 100; // Number of games to simulate for training data
        generateCSVData(trainingDataFile, numGames);

        // Load the training data from the CSV file into a Smile dataset (Anh code)
        List<double[]> values = new ArrayList<>();
        List<Integer> intLabels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
            String line = br.readLine(); 
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] row = new double[parts.length - 1];
                for (int i = 0; i < row.length; i++) {
                    row[i] = Double.parseDouble(parts[i]);
                }
                values.add(row);
                intLabels.add(Integer.parseInt(parts[parts.length - 1]));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // -----------------------------
        // 1) Your data
        // -----------------------------

        double[][] X = values.toArray(new double[0][]);
        int[] y = intLabels.stream().mapToInt(i -> i).toArray();

        int nSamples  = X.length;
        int nFeatures = (nSamples == 0 ? 0 : X[0].length);

        // Defensive checks
        if (nSamples == 0 || nFeatures == 0 || y.length != nSamples) {
            throw new IllegalArgumentException("Bad shapes: X=" + nSamples + "x" + nFeatures + ", y.length=" + y.length);
        }


        // -----------------------------------
        // 2) Wrap as INDArrays / DataSet
        // -----------------------------------
        // Ensure floating dtype for both features and labels
        INDArray features = Nd4j.createFromArray(X).castTo(Nd4j.defaultFloatingPointType());
        // y -> column vector [nSamples,1], cast to same dtype as features
        INDArray labels   = Nd4j.createFromArray(y).reshape(nSamples, 1).castTo(features.dataType());

        DataSet all = new DataSet(features, labels);
        // // Optional (often helpful): standardize features; need to save and reuse if done
        // DataNormalization norm = new NormalizerStandardize();
        // norm.fit(all);
        // norm.transform(all);

        // Shuffle and batch
        System.out.printf("Training data: %d samples, %d features%n", nSamples, nFeatures);
        long seed = 123;
        all.shuffle(seed);
        int batchSize = Math.min(128, nSamples);
        DataSetIterator trainIter = new ListDataSetIterator<>(all.asList(), batchSize);

        // -----------------------------------
        // 3) Define the network
        // -----------------------------------
        // One hidden layer, output: 1 unit with sigmoid, XENT loss
        int nHidden = 32;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))                    // optimizer & LR
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(nFeatures)
                        .nOut(nHidden)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(nHidden)
                        .nOut(1)
                        .activation(Activation.SIGMOID)     // predict P(y=1|x)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
//        model.setListeners(new ScoreIterationListener(1000));

        // -----------------------------------
        // 4) Train
        // -----------------------------------
        int epochs = 20;

        // Create a validation dataset iterator (e.g., using a split from the original dataset)
        int validationSize = nSamples / 10; // Use 10% of the data for validation
        DataSetIterator valIter = new ListDataSetIterator<>(all.asList().subList(0, validationSize), validationSize);

        for (int i = 0; i < epochs; i++) {
            trainIter.reset();
            model.fit(trainIter);
            // at epoch end:
            double trainLoss = new DataSetLossCalculator(trainIter, true).calculateScore(model);
            double valLoss   = new DataSetLossCalculator(valIter,   true).calculateScore(model);
            System.out.printf("epoch %d  trainLoss=%.5f  valLoss=%.5f%n",
            model.getEpochCount(), trainLoss, valLoss);
        }

        // -----------------------------------
        // 5) Inference: probabilities P(y=1|x)
        // -----------------------------------
        INDArray probs = model.output(features, false);  // shape: [nSamples,1], sigmoid outputs in [0,1]

        // Example: print first 10 probabilities and hard predictions (threshold 0.5)
        for (int i = 0; i < Math.min(10, nSamples); i++) {
            double p = probs.getDouble(i);
            int pred = (p >= 0.5) ? 1 : 0;
            System.out.printf("i=%d  y=%d  p=%.4f  pred=%d%n", i, y[i], p, pred);
        }


        try {
            ModelSerializer.writeModel(model, new java.io.File(modelFilename), true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // // Save the model to a file using an ObjectOutputStream
        // try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(modelFilename))) {
        //     oos.writeObject(model);
        // } catch (IOException e) {
        //     e.printStackTrace();
        // }


        // Delete the training data file after learning the model
        java.nio.file.Path path = java.nio.file.Paths.get(trainingDataFile);
        try {
            java.nio.file.Files.delete(path);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double predict(SPState state) {
        // Create a double array for the feature values
        double[] featureValues = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object value = features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }

        // Predict using the neural network model
        // From the featureValues, create an input matrix with one row for the model
        INDArray features = Nd4j.createFromArray(new double[][] {featureValues}).castTo(Nd4j.defaultFloatingPointType());
        return model.output(features, false).getDouble(0);
    }

    // min_deck_size – the number of cards in the smallest phase deck
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

    // points – current player points
    class SPFeaturePoints extends SPFeature {
        public SPFeaturePoints() {
            super("points", "current player points");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn];
        }
    }

    // points_diff – current player points relative to the opponent (assumes two players)
    class SPFeaturePointsDiff extends SPFeature {
        public SPFeaturePointsDiff() {
            super("points_diff", "current player points relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn] - state.playerPoints[1 - state.playerTurn];
        }
    }

    // rubles – current player rubles (money)
    class SPFeatureRubles extends SPFeature {
        public SPFeatureRubles() {
            super("rubles", "current player rubles (money)");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn];
        }
    }

    // rubles_diff – current player rubles (money) relative to the opponent
    class SPFeatureRublesDiff extends SPFeature {
        public SPFeatureRublesDiff() {
            super("rubles_diff", "current player rubles (money) relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn] - state.playerRubles[1 - state.playerTurn];
        }
    }

    // points_round_gain – the number of points the current player is gaining per round
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

    // points_round_gain_diff – the number of points the current player is gaining per round relative to the opponent
    class SPFeaturePointsRoundGainDiff extends SPFeature {
        public SPFeaturePointsRoundGainDiff() {
            super("points_round_gain_diff", "the number of points the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
            int opponentPointsPerRound = state.playerWorkers.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum();
            return pointsPerRound - opponentPointsPerRound;
        }
    }

    // rubles_round_gain – the number of rubles the current player is gaining per round
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

    // rubles_round_gain_diff – the number of rubles the current player is gaining per round relative to the opponent
    class SPFeatureRublesRoundGainDiff extends SPFeature {
        public SPFeatureRublesRoundGainDiff() {
            super("rubles_round_gain_diff", "the number of rubles the current player is gaining per round relative to the opponent");
        }

        public Object getValue(SPState state) {
            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            int opponentRublesPerRound = state.playerWorkers.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
            return rublesPerRound - opponentRublesPerRound;
        }
    }

    // unique_aristocrats – the number of unique aristocrats of the current player
    class SPFeatureUniqueAristocrats extends SPFeature {
        public SPFeatureUniqueAristocrats() {
            super("unique_aristocrats", "the number of unique aristocrats of the current player");
        }

        public Object getValue(SPState state) {
            return state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
        }
    }

    // unique_aristocrats_diff – the number of unique aristocrats of the current player relative to the opponent
    class SPFeatureUniqueAristocratsDiff extends SPFeature {
        public SPFeatureUniqueAristocratsDiff() {
            super("unique_aristocrats_diff", "the number of unique aristocrats of the current player relative to the opponent");
        }

        public Object getValue(SPState state) {
            long uniqueAristocrats = state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
            long opponentUniqueAristocrats = state.playerAristocrats.get(1 - state.playerTurn).stream().distinct().count();
            return uniqueAristocrats - opponentUniqueAristocrats;
        }
    }

    // cards_in_hand – the number of cards in the current player hand
    class SPFeatureCardsInHand extends SPFeature {
        public SPFeatureCardsInHand() {
            super("cards_in_hand", "the number of cards in the current player hand");
        }

        public Object getValue(SPState state) {
            return state.playerHands.get(state.playerTurn).size();
        }
    }

    // cards_in_hand_diff – the number of cards in the current player hand relative to the opponent
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
     * Estimates expected points and divides by estimated remaining turns (averaged).
     */
    class ROIFeature extends SPFeature {
        private final int samples = 50;   // number of playouts for round estimate
        private final int maxDepth = 30;  // cap for playout length

        public ROIFeature() {
            super("ROI", "estimated points per remaining round");
        }

        @Override
        public Object getValue(SPState state) {
            double expectedPoints = estimatePoints(state);
            double estRoundsLeft = estimateRounds(state);
            return expectedPoints / Math.max(1.0, estRoundsLeft);
        }

        private double estimateRounds(SPState root) {
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
            return (double) totalTurns / (double) samples;
        }

        private double estimatePoints(SPState state) {
            int player = state.playerTurn;
            int currentPoints = state.playerPoints[player];

            // best single-card points we could buy immediately
            int bestCardPoints = 0;
            for (SPAction action : state.getLegalActions()) {
                if (action instanceof SPBuyAction) {
                    SPCard card = ((SPBuyAction) action).card;
                    if (card != null) bestCardPoints = Math.max(bestCardPoints, card.points);
                }
            }

            // unique aristocrats owned by player
            Set<String> uniqueAris = new HashSet<>();
            ArrayList<SPCard> myAris = state.playerAristocrats.get(player);
            for (SPCard c : myAris) {
                if (c != null && c.isAristocrat) uniqueAris.add(c.name);
            }
            int n = uniqueAris.size();
            int aristocratBonus = 0;
            if (n > 0) {
                if (SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS != null && !SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.isEmpty()) {
                    aristocratBonus = SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.get(Math.min(n, SPState.UNIQUE_ARISTOCRAT_BONUS_POINTS.size() - 1));
                } else {
                    int[] bonusTable = {0, 1, 3, 6, 10, 15, 21, 28, 36};
                    aristocratBonus = bonusTable[Math.min(n, bonusTable.length - 1)];
                }
            }

            return currentPoints + bestCardPoints + aristocratBonus;
        }
    }

    // Inside OKStateFeaturesRF1.java, after your other feature classes:
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

    class DontBeAnIdiotFeature extends SPFeature{
    public DontBeAnIdiotFeature(){
        super("Dont_Be_Idiot", "Penalizes buying expensive buildings in round 1");
    }

    @Override
    public Object getValue(SPState state){
        // Condition 1: Must be in the very beginning of the game (Round 1)
        if (state.round != 1) {
            return 0.0;
        }

        // Condition 2: Check if the player owns any expensive buildings (cost >= 8)
        // If the player has one, it means they bought it in round 1
        int player = state.playerTurn;
        for (SPCard building : state.playerBuildings.get(player)) {
            // Assuming an "expensive" building is anything costing 8 rubles or more
            if (building.cost >= 8) {
                // If this state resulted from buying an expensive building in round 1, penalize it
                return -1.0; 
            }
        }
        // Good play detected (or action wasn't a buy): no penalty
        return 0.0;
    }
}

    class ReserveRublesFeature extends SPFeature {
        public ReserveRublesFeature() {
            super("Reserve_Rubles_Check", "Has at least 10 rubles before the Worker phase");
        }

        @Override
        public Object getValue(SPState state) {
            // This feature should check the state *before* the income phase happens
            // The Worker phase is the income phase (SPState.WORKER = 0)
            if (state.phase == SPState.WORKER) {
                int playerRubles = state.playerRubles[state.playerTurn];
                
                // Check if the reserve is met
                if (playerRubles >= 10) {
                    return 1.0; // Meets the goal, good state
                } else {
                    return 0.0; // Fails the goal, neutral or slightly bad state
                }
            }
            
            // If it's not the Worker phase, this check is less relevant, so we return a neutral value
            return 0.0;
        }
    }

    /**
     * SPFeatureStrategicBuildingPreference - Prioritizes Warehouse > Observatory, 
     * and dynamically rewards Pubs for their end-game VP efficiency.
     */
    class SPFeatureStrategicBuildingPreference extends SPFeature {
        public SPFeatureStrategicBuildingPreference() {
            super("Strategic_Building_Pref", "Dynamic utility card preference with phase awareness");
        }

        @Override
        public Object getValue(SPState state) {
            double score = 0.0;
            int player = state.playerTurn;
            int numObservatories = 0;
            int numPubs = 0;
            boolean hasWarehouse = false;
            
            // --- 1. Calculate Player Metrics ---
            for (SPCard card : state.playerBuildings.get(player)) {
                if (card.name.equals("Observatory")) {
                    numObservatories++;
                } else if (card.name.equals("Pub")) {
                    numPubs++;
                } else if (card.name.equals("Warehouse")) {
                    hasWarehouse = true;
                }
            }
            
            // --- 2. Implement Warehouse Priority (Highest Utility/Early Game) ---
            // If we don't have a Warehouse and are in the income-building phase (early rounds)
            if (!hasWarehouse && state.round <= 4) {
                // Large bonus to encourage buying the Warehouse early
                score += 2.0; 
            } else if (hasWarehouse) {
                // Small enduring bonus for having the key utility
                score += 0.3;
            }
            
            // --- 3. Implement Observatory Utility (Mid-Game Card Flow) ---
            // Observatories are always good for hand size and card flow.
            score += 0.7 * numObservatories;
            
            // --- 4. Implement Pub Dynamic VP/End-Game Push ---
            
            // The value of a Pub is heavily dependent on the game phase.
            double pubValueMultiplier = 0.0;
            if (state.round <= 3) {
                // EARLY GAME: Pub is a distraction from income building. Apply minor penalty for the tax.
                pubValueMultiplier = -0.3; 
            } else if (state.round <= 6) {
                // MID GAME: Pub is okay, but not priority. Low score, neutral.
                pubValueMultiplier = 0.1; 
            } else { // round 7 or later (Late Game/End Game)
                // LATE GAME: Pub is a crucial VP conversion engine.
                // It provides 1 VP for 2 Rubles (high efficiency), making it very valuable.
                pubValueMultiplier = 1.5; 
            }

            score += pubValueMultiplier * numPubs; 
            
            // --- 5. Apply the Minor Recurring Tax Penalty ---
            // The Pub has a 1-ruble tax. This tax slightly reduces the score regardless of phase.
            int pubTax = numPubs * 1; 
            score -= 0.1 * pubTax; 

            // Scale the final score down slightly for stability in the LR model
            return score / 3.0; 
        }
    }

    public static void main(String[] args) {
        new SPStateFeaturesNN1();
    }

}
