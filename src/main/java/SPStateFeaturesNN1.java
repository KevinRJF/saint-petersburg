import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

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


public class SPStateFeaturesNN1 {
    String modelFilename = "SPNN1.model";
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

    public SPStateFeaturesNN1() {
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
                SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPRandomPlayer(), new SPRandomPlayer());
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

        String trainingDataFile = "SPTrainingData.csv";
        int numGames = 5000; // Number of games to simulate for training data
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

    public static void main(String[] args) {
        new SPStateFeaturesNN1();
    }

}
