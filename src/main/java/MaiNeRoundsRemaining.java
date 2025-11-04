import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MaiNeRoundsRemaining {

    String modelFilename = "MaiNeRoundsRemaining.model";
    private MultiLayerNetwork model;
    ArrayList<SPFeature> features;
    private NormalizerStandardize normalizer; 

    File modelFile = new File(modelFilename);

    public void saveModel() {
        try {
            ModelSerializer.writeModel(model, modelFile, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public ArrayList<Object> getFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : features) {
            values.add(feature.getValue(state));
        }
        return values;
    }

    
    public String getCSVHeader() {
        StringBuilder header = new StringBuilder();
        for (SPFeature feature : features) {
            header.append(feature.getName()).append(",");
        }
        header.append("rounds_remaining"); // rounds remaining including the current round
        return header.toString();
    }

    public String getCSVRow(SPState state, int totalRounds) {
        int roundsRemaining = totalRounds - state.round; 
        StringBuilder row = new StringBuilder();
        for (SPFeature feature : features) {
            row.append(feature.getValue(state)).append(",");
        }
        row.append(roundsRemaining);
        return row.toString();
    }

    public String getCSVRows(SPGameTranscript transcript) {
        StringBuilder rows = new StringBuilder();
        int totalRounds = transcript.states.getLast().round;
        for (SPState state : transcript.getStates()) {
            rows.append(getCSVRow(state, totalRounds)).append("\n");
        }
        return rows.toString();
    }
    
    public void generateCSVData(String filename, int numGames) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println(getCSVHeader());
            SPPlayerFlatMC player1 = new SPPlayerFlatMC();
            player1.numSimulationsPerAction = 100;
            //player1.playoutTerminationDepth = 2;
            MaiNePlayer player2 = new MaiNePlayer();
            player2.numSimulationsPerAction = 100;
            //player2.playoutTerminationDepth = 2;

            long startTime = System.currentTimeMillis();
            for (int i = 0; i < numGames; i++) {
                SPGameTranscript transcript = SPSimulateGame.simulateGame(player1, player2);
                writer.print(getCSVRows(transcript));
            }
            long endTime = System.currentTimeMillis();
            System.out.println("Generated " + numGames + " games in " + (endTime - startTime) / 1000.0 + " seconds.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public MaiNeRoundsRemaining() {
        System.out.println("Starting MaiNeRoundsRemaining ...");
        features = new ArrayList<>();
        features.add(new SPFeatureCurrentRound());
        features.add(new SPFeatureCurrentPhase());
        for (int phase = 0; phase < 4; phase++) {
            features.add(new SPFeatureCardsInDeck(phase));
            if (phase < 3)
                for (int player = 0; player < 2; player++) {
                    features.add(new SPFeaturePointsPerPhase(phase, player));
                    features.add(new SPFeatureRublesPerPhase(phase, player));
                }
        }
        features.add(new SPFeatureMinDeckSize());
        for (int player = 0; player < 2; player++) {
            features.add(new SPFeaturePlayerNPoints(player));
            features.add(new SPFeaturePlayerNRubles(player));
            features.add(new SPFeatureUniqueAristocratPoints(player));
            features.add(new SPFeatureDuplicateAristocrats(player));
            features.add(new SPFeatureCardsInHandN(player));
        }

        features.add(new SPFeatureCardsOfferedTop());
        features.add(new SPFeatureCardsOfferedBottom());

        // generate CSV play data
        //generateCSVData("rounds_remaining_test.csv", 1000);  

        //generateCSVData("rounds_remaining.csv", 1000);
        initializeModel();
    }

    /**
     * Correctly loads the model AND the normalizer.
     */
    private void initializeModel() {

        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file does not exist. Generating model...");
            learnModel();
        }

        try {
            Pair<MultiLayerNetwork, Normalizer> pair = 
                ModelSerializer.restoreMultiLayerNetworkAndNormalizer(modelFile, true);
    
            model = pair.getFirst();
            this.normalizer = (NormalizerStandardize) pair.getSecond();
    
        } catch (IOException e) {
            e.printStackTrace();
            // handle exception
        }
    }

    // ... (CSV methods are unchanged and omitted for brevity) ...
    // NOTE: Keep all your CSV methods here!
    
    // ... (generateCSVData method is unchanged and omitted for brevity) ...

    /**
     * Performs model training, fits the normalizer, and saves both.
     */
    public void learnModel() {
        String trainingDataFile = "rounds_remaining.csv";
        List<double[]> Xlist = new ArrayList<>();
        List<Double> ylist = new ArrayList<>();

        // ... (Code to load data from CSV into Xlist and ylist - kept as is) ...
        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
            br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] row = new double[parts.length - 1];
                for (int i = 0; i < row.length; i++) row[i] = Double.parseDouble(parts[i]);
                Xlist.add(row);
                ylist.add(Double.parseDouble(parts[parts.length - 1]));
            }
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        int nSamples = Xlist.size();
        int nFeatures = Xlist.get(0).length;

        double[][] X = Xlist.toArray(new double[0][]);
        double[] y = ylist.stream().mapToDouble(d -> d).toArray();

        INDArray featuresArray = Nd4j.createFromArray(X);
        INDArray labelsArray = Nd4j.createFromArray(y).reshape(nSamples, 1);

        DataSet allData = new DataSet(featuresArray, labelsArray); // FIX: renamed 'all' to 'allData'
        allData.shuffle(123);

        normalizer = new NormalizerStandardize(); // Uses the class member
        normalizer.fit(allData); // compute mean and std
        normalizer.transform(allData); // apply normalization

        int nHidden = 32;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(1e-3))
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder().nIn(nFeatures).nOut(nHidden).activation(Activation.RELU).build())
            .layer(new DenseLayer.Builder().nIn(nHidden).nOut(nHidden/2).activation(Activation.RELU).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(nHidden/2).nOut(1)
                .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        // FIX: The model variable inside learnModel() should be the class member for saving
        this.model = model; 

        int batchSize = Math.min(128, nSamples);
        DataSetIterator trainIter = new ListDataSetIterator<>(allData.asList(), batchSize);

        int epochs = 50;
        System.out.println("\nStarting training for " + epochs + " epochs...");
        for (int i = 0; i < epochs; i++) {
            trainIter.reset();
            this.model.fit(trainIter);

            trainIter.reset();
            // RegressionEvaluation eval = this.model.evaluate(trainIter); // Can use evaluate() if up-to-date
            RegressionEvaluation eval = this.model.evaluateRegression(trainIter);
            System.out.printf("Epoch %2d | MAE=%.4f | MSE=%.4f | RMSE=%.4f%n",
                i + 1,
                eval.meanAbsoluteError(0),
                eval.meanSquaredError(0),
                eval.rootMeanSquaredError(0));
        }
        
        try {
            ModelSerializer.writeModel(
                this.model, 
                new java.io.File(modelFilename), 
                true, 
                normalizer 
            );
            System.out.println("\nModel and Normalizer saved to " + modelFilename);
        } catch (IOException e) {
            System.err.println("Error saving the model: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Performs a prediction on a new game state using the loaded model and normalizer.
     */
    public double predictRoundsRemaining(SPState state) {
        if (model == null) {
            System.err.println("Error: Model is not initialized or loaded.");
            return Double.NaN; 
        }
    
        // 1. Extract Features
        ArrayList<Object> featureValues = getFeatureValues(state);
        double[] inputFeatures = new double[featureValues.size()];
    
        // Convert List<Object> to double[]
        for (int i = 0; i < featureValues.size(); i++) {
            inputFeatures[i] = ((Number) featureValues.get(i)).doubleValue();
        }
    
        // 2. Create INDArray for prediction
        INDArray input = Nd4j.createFromArray(inputFeatures).reshape(1, inputFeatures.length);
    
        // 3. Apply the *saved* normalizer
        if (normalizer != null) {
            normalizer.transform(input); 
        } else {
            System.err.println("Warning: Normalizer not loaded. Prediction accuracy compromised.");
        }
    
        // 4. Get the model's prediction and return
        INDArray output = model.output(input, false);
        return Math.max(0.0, output.getDouble(0)); 
    }
    
    //  FEATURES 
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
    
    // current_round - current round number
    class SPFeatureCurrentRound extends SPFeature {
        public SPFeatureCurrentRound() {
            super("current_round", "current round number");
        }

        public Object getValue(SPState state) {
            return state.round;
        }
    }

    // current_phase - current phase number
    class SPFeatureCurrentPhase extends SPFeature {
        public SPFeatureCurrentPhase() {
            super("current_phase", "current phase number");
        }

        public Object getValue(SPState state) {
            return state.phase;
        }
    }

    // cards_in_deck_# – number of cards remaining in phase # deck
    class SPFeatureCardsInDeck extends SPFeature {
        private int phase;

        public SPFeatureCardsInDeck(int phase) {
            super("cards_in_deck_" + phase, "number of cards remaining in phase " + phase + " deck");
            this.phase = phase;
        }

        public Object getValue(SPState state) {
            if (phase == 0) {
                return state.workerDeck.size();
            } else if (phase == 1) {
                return state.buildingDeck.size();
            } else if (phase == 2) {
                return state.aristocratDeck.size();
            } else {
                return state.tradingDeck.size();
            }
        }
    }       

    // player_#_points – points of player #
    class SPFeaturePlayerPoints extends SPFeature {
        private int playerIndex;

        public SPFeaturePlayerPoints(int playerIndex) {
            super("player_" + playerIndex + "_points", "points of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            return state.playerPoints[playerIndex];
        }
    }

    // points_phase_#_player_# - points of player # in phase #
    class SPFeaturePointsPerPhase extends SPFeature {
        private int phase;
        private int player;

        public SPFeaturePointsPerPhase(int phase, int player) {
            super("points_phase_" + phase + "_player_" + player, "points of player " + player + " in phase " + phase);
            this.phase = phase;
            this.player = player;
        }

        public Object getValue(SPState state) {
            ArrayList<SPCard> cards;
            if (phase == 0) {
                cards = state.playerWorkers.get(player);
            } else if (phase == 1) {
                cards = state.playerBuildings.get(player);
            } else {
                cards = state.playerAristocrats.get(player);
            }
            return cards.stream().mapToInt(card -> card.points).sum();
        }
    }

    // rubles_phase_#_player_# - rubles of player # in phase #
    class SPFeatureRublesPerPhase extends SPFeature {
        private int phase;
        private int player;

        public SPFeatureRublesPerPhase(int phase, int player) {
            super("rubles_phase_" + phase + "_player_" + player, "rubles of player " + player + " in phase " + phase);
            this.phase = phase;
            this.player = player;
        }

        public Object getValue(SPState state) {
            ArrayList<SPCard> cards;
            if (phase == 0) {
                cards = state.playerWorkers.get(player);
            } else if (phase == 1) {
                cards = state.playerBuildings.get(player);
            } else {
                cards = state.playerAristocrats.get(player);
            }
            return cards.stream().mapToInt(card -> card.rubles).sum();
        }
    }

    // cards_in_hand_# – number of cards in hand of player #
    class SPFeatureCardsInHandN extends SPFeature {
        private int playerIndex;

        public SPFeatureCardsInHandN(int playerIndex) {
            super("cards_in_hand_" + playerIndex, "number of cards in hand of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            return state.playerHands.get(playerIndex).size();
        }
    }

    // player_#_rubles – rubles (money) of player #
    class SPFeaturePlayerNRubles extends SPFeature {
        private int playerIndex;

        public SPFeaturePlayerNRubles(int playerIndex) {
            super("player_" + playerIndex + "_rubles", "rubles (money) of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            return state.playerRubles[playerIndex];
        }
    }

    // player_#_points – points of player #
    class SPFeaturePlayerNPoints extends SPFeature {
        private int playerIndex;

        public SPFeaturePlayerNPoints(int playerIndex) {
            super("player_" + playerIndex + "_points", "points of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            return state.playerPoints[playerIndex];
        }
    }

    // player_#_unique_aristocrat_points – points of unique aristocrats of the current player
    class SPFeatureUniqueAristocratPoints extends SPFeature {
        private int playerIndex;

        public SPFeatureUniqueAristocratPoints(int playerIndex) {
            super("player_" + playerIndex + "_unique_aristocrat_points", "points of unique aristocrats of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            int uniqueAristocrats = state.playerAristocrats.get(playerIndex).stream().distinct().mapToInt(card -> card.points).sum();
            return (uniqueAristocrats * (uniqueAristocrats + 1) / 2);
        }
    }

    // player_#_duplicate_aristocrats – number of duplicate aristocrats of the current player
    class SPFeatureDuplicateAristocrats extends SPFeature {
        private int playerIndex;

        public SPFeatureDuplicateAristocrats(int playerIndex) {
            super("player_" + playerIndex + "_duplicate_aristocrats", "number of duplicate aristocrats of player " + playerIndex);
            this.playerIndex = playerIndex;
        }

        public Object getValue(SPState state) {
            Set<SPCard> uniqueAristocrats = new HashSet<>(state.playerAristocrats.get(playerIndex));
            return state.playerAristocrats.get(playerIndex).size() - uniqueAristocrats.size();
        }
    }

    // cards_offered_top – number of cards offered on the top of the deck
    class SPFeatureCardsOfferedTop extends SPFeature {
        public SPFeatureCardsOfferedTop() {
            super("cards_offered_top", "number of cards offered on the top of the deck");
        }

        public Object getValue(SPState state) {
            return state.upperCardRow.size();
        }
    }

    // cards_offered_bottom – number of cards offered on the bottom of the deck
    class SPFeatureCardsOfferedBottom extends SPFeature {
        public SPFeatureCardsOfferedBottom() {
            super("cards_offered_bottom", "number of cards offered on the bottom of the deck");
        }

        public Object getValue(SPState state) {
            return state.lowerCardRow.size();
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

    public void testModel(String testDataFile) {
    List<double[]> Xlist = new ArrayList<>();
    List<Double> ylist = new ArrayList<>();

    // Load test data CSV (same format as training)
    try (BufferedReader br = new BufferedReader(new FileReader(testDataFile))) {
        br.readLine(); // skip header
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            double[] row = new double[parts.length - 1];
            for (int i = 0; i < row.length; i++) row[i] = Double.parseDouble(parts[i]);
            Xlist.add(row);
            ylist.add(Double.parseDouble(parts[parts.length - 1]));
        }
    } catch (IOException e) {
        e.printStackTrace();
        return;
    }

    int nSamples = Xlist.size();
    int nFeatures = Xlist.get(0).length;

    double[][] X = Xlist.toArray(new double[0][]);
    double[] y = ylist.stream().mapToDouble(d -> d).toArray();

    // Create INDArray from features and labels
    INDArray featuresArray = Nd4j.createFromArray(X);
    INDArray labelsArray = Nd4j.createFromArray(y).reshape(nSamples, 1);

    DataSet testData = new DataSet(featuresArray, labelsArray);

    // Normalize test data with the same normalizer used in training
    normalizer.transform(testData);

    // Run predictions
    INDArray predicted = model.output(testData.getFeatures());

    // Evaluate regression metrics
    RegressionEvaluation eval = new RegressionEvaluation(1);
    eval.eval(testData.getLabels(), predicted);

    // System.out.printf("Test Results | MAE=%.4f | MSE=%.4f | RMSE=%.4f%n",
    //         eval.meanAbsoluteError(0),
    //         eval.meanSquaredError(0),
    //         eval.rootMeanSquaredError(0));
    }

    public static void main(String[] args) {
        new MaiNeRoundsRemaining();
    }
}