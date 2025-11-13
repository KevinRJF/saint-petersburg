import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

// DL4J / ND4J imports (neural network)
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;

public class OKBasedFeatures {

    String modelFilename = "OKNN.model";
    ArrayList<SPFeature> features;
    private MultiLayerNetwork model;

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
                SPGameTranscript transcript = SPSimulateGame.simulateGame(new OKTurnBasedFeaturesPlayer(), new OKTurnBasedFeaturesPlayer());
                writer.print(getCSVRows(transcript));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void learnModel() {
        String trainingDataFile = "SPTrainingDataNN.csv";
        int numGames = 200;
        generateCSVData(trainingDataFile, numGames);

        List<double[]> values = new ArrayList<>();
        List<Integer> intLabels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
            String line = br.readLine(); // header
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

        double[][] X = values.toArray(new double[0][]);
        int[] y = intLabels.stream().mapToInt(i -> i).toArray();

        int nSamples = X.length;
        int nFeatures = (nSamples == 0 ? 0 : X[0].length);
        if (nSamples == 0 || nFeatures == 0 || y.length != nSamples) {
            throw new IllegalArgumentException("Bad shapes: X=" + nSamples + "x" + nFeatures + ", y.length=" + y.length);
        }

        INDArray featuresArr = Nd4j.createFromArray(X).castTo(Nd4j.defaultFloatingPointType());
        INDArray labelsArr = Nd4j.createFromArray(y).reshape(nSamples, 1).castTo(featuresArr.dataType());

        DataSet all = new DataSet(featuresArr, labelsArr);
        all.shuffle(123);
        int batchSize = Math.min(128, nSamples);
        DataSetIterator trainIter = new ListDataSetIterator<>(all.asList(), batchSize);

        int nHidden = Math.max(16, nFeatures * 2);
        long seed = 123;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(nFeatures).nOut(nHidden).activation(Activation.SIGMOID).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(nHidden).nOut(1).activation(Activation.SIGMOID).build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();

        int epochs = 20;
        int validationSize = Math.max(1, nSamples / 10);
        DataSetIterator valIter = new ListDataSetIterator<>(all.asList().subList(0, validationSize), validationSize);

        for (int i = 0; i < epochs; i++) {
            trainIter.reset();
            model.fit(trainIter);
            double trainLoss = new DataSetLossCalculator(trainIter, true).calculateScore(model);
            double valLoss = new DataSetLossCalculator(valIter, true).calculateScore(model);
            System.out.printf("epoch %d  trainLoss=%.5f  valLoss=%.5f%n", model.getEpochCount(), trainLoss, valLoss);
        }

        try {
            ModelSerializer.writeModel(model, new java.io.File(modelFilename), true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            java.nio.file.Files.delete(java.nio.file.Paths.get(trainingDataFile));
        } catch (IOException e) {
            // ignore
        }
    }

    public double predict(SPState state) {
        double[] featureValues = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object value = features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }

        INDArray input = Nd4j.createFromArray(new double[][] {featureValues}).castTo(Nd4j.defaultFloatingPointType());
        if (model == null) {
            // fallback heuristic: average normalized features
            double sum = 0.0;
            for (double v : featureValues) sum += v;
            return 1.0 / (1.0 + Math.exp(- (sum / Math.max(1, featureValues.length))));
        }
        return model.output(input, false).getDouble(0);
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