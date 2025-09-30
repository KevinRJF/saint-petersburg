//import java.io.BufferedReader;
//import java.io.FileNotFoundException;
//import java.io.FileReader;
//import java.io.FileWriter;
//import java.io.IOException;
//import java.io.PrintWriter;
//import java.util.ArrayList;
//import java.util.List;
//
//import smile.classification.LogisticRegression;
//
//public class SPStateFeaturesLR2 {
//    String modelFilename = "SPLogisticRegression2.model";
//    LogisticRegression.Binomial model;
//    ArrayList<SPFeature> features;
//
//    public ArrayList<Object> getFeatureValues(SPState state) {
//        ArrayList<Object> values = new ArrayList<>();
//        for (SPFeature feature : features) {
//            values.add(feature.getValue(state));
//        }
//        return values;
//    }
//
//    public SPStateFeaturesLR2() {
//        features = new ArrayList<>();
//        features.add(new SPFeatureMinDeckSize());
//        features.add(new SPFeaturePoints());
//        features.add(new SPFeatureInteractionTerm(new SPFeaturePoints(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeaturePointsDiff());
//        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsDiff(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeatureRubles());
//        features.add(new SPFeatureInteractionTerm(new SPFeatureRubles(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeatureRublesDiff());
//        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesDiff(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeaturePointsRoundGain());
//        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGain(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeaturePointsRoundGainDiff());
//        features.add(new SPFeatureInteractionTerm(new SPFeaturePointsRoundGainDiff(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeatureRublesRoundGain());
//        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGain(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeatureRublesRoundGainDiff());
//        features.add(new SPFeatureInteractionTerm(new SPFeatureRublesRoundGainDiff(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeatureUniqueAristocrats());
//        features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocrats(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeatureUniqueAristocratsDiff());
//        features.add(new SPFeatureInteractionTerm(new SPFeatureUniqueAristocratsDiff(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeatureCardsInHand());
//        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHand(), new SPFeatureMinDeckSize()));
//        features.add(new SPFeatureCardsInHandDiff());
//        features.add(new SPFeatureInteractionTerm(new SPFeatureCardsInHandDiff(), new SPFeatureMinDeckSize()));
////        features.add(new SPFeatureAristocratsInHandsDiff());
////        features.add(new SPFeatureInteractionTerm(new SPFeatureAristocratsInHandsDiff(), new SPFeatureMinDeckSize()));
//
//        initializeModel();
//    }
//
//    private void initializeModel() {
//        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
//            System.out.println("Model file does not exist. Generating model...");
//            learnModel();
//        }
//        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(new java.io.FileInputStream(modelFilename))) {
//            model = (LogisticRegression.Binomial) ois.readObject();
//        } catch (IOException | ClassNotFoundException e) {
//            e.printStackTrace();
//        }
//    }
//
//    public String getCSVHeader() {
//        StringBuilder header = new StringBuilder();
//        for (SPFeature feature : features) {
//            header.append(feature.getName()).append(",");
//        }
//        header.append("is_winner");
//        return header.toString();
//    }
//
//    public String getCSVRow(SPState state, boolean[] isWinner) {
//        int currentPlayerIndex = state.playerTurn;
//        int winnerVal = isWinner[currentPlayerIndex] ? 1 : 0;
//        StringBuilder row = new StringBuilder();
//        for (SPFeature feature : features) {
//            row.append(feature.getValue(state)).append(",");
//        }
//        row.append(winnerVal);
//        return row.toString();
//    }
//
//    public String getCSVRows(SPGameTranscript transcript) {
//        StringBuilder rows = new StringBuilder();
//        boolean[] isWinner = transcript.getWinners();
//        for (SPState state : transcript.getStates()) {
//            rows.append(getCSVRow(state, isWinner)).append("\n");
//        }
//        return rows.toString();
//    }
//
//    public void generateCSVData(String filename, int numGames) {
//        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
//            writer.println(getCSVHeader());
//            for (int i = 0; i < numGames; i++) {
//                // SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPRandomPlayer(), new SPRandomPlayer());
//            	 // SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPMMustafizPlayer(), new SPMMustafizPlayer());
//                 SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPPlayerFlatMC(), new SPPlayerFlatMC());
//                writer.print(getCSVRows(transcript));
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
//
//    public void learnModel() {
//        // This method assumes that the logistic regression model has not been created and saved yet.
//        // It generates training data by simulating games and saves it to a CSV file.
//        // Then it uses logistic regression to learn a model and saves it to a file.
//    	try {
//            java.nio.file.Files.deleteIfExists(java.nio.file.Paths.get(modelFilename));
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    	
//        String trainingDataFile = "SPTrainingData.csv";
//        int numGames = 1000; // Number of games to simulate for training data
//        generateCSVData(trainingDataFile, numGames);
//
//        // Load the training data from the CSV file into a Smile dataset (Anh code)
//        List<double[]> values = new ArrayList<>();
//        List<Integer> labels = new ArrayList<>();
//
//        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
//            String line = br.readLine(); 
//            while ((line = br.readLine()) != null) {
//                String[] parts = line.split(",");
//                double[] row = new double[parts.length - 1];
//                for (int i = 0; i < row.length; i++) {
//                    row[i] = Double.parseDouble(parts[i]);
//                }
//                values.add(row);
//                labels.add(Integer.parseInt(parts[parts.length - 1]));
//            }
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        double[][] X = values.toArray(new double[0][]);
//        int[] y = labels.stream().mapToInt(i -> i).toArray();
//
//        // Perform logistic regression using the Smile library
//        LogisticRegression.Binomial model = LogisticRegression.binomial(X, y);
//
//        // Save the model to a file using an ObjectOutputStream
//        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(modelFilename))) {
//            oos.writeObject(model);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        // Print the model coefficients along with their feature names
//        System.out.println("Model coefficients:");
//        System.out.println(features.size() + " features");
//        System.out.println(model.coefficients().length + " coefficients");
//        System.out.printf("%.4f\tIntercept%n", model.coefficients()[0]);
//        for (int i = 0; i < model.coefficients().length - 1; i++) {
//            System.out.printf("%.4f\t%s%n", model.coefficients()[i + 1], features.get(i).getName());
//        }
//
//        // Delete the training data file after learning the model
//        java.nio.file.Path path = java.nio.file.Paths.get(trainingDataFile);
//        try {
//            java.nio.file.Files.delete(path);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
//
//    public double predict(SPState state) {
//        // Create a double array for the feature values
//        double[] featureValues = new double[features.size()];
//        for (int i = 0; i < features.size(); i++) {
//            Object value = features.get(i).getValue(state);
//            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
//        }
//
//        // Use the logistic regression model to predict the probability of winning
//        return model.score(featureValues);
//    }
//
//    // min_deck_size – the number of cards in the smallest phase deck
//    class SPFeatureMinDeckSize extends SPFeature {
//        public SPFeatureMinDeckSize() {
//            super("min_deck_size", "the number of cards in the smallest phase deck");
//        }
//
//        public Object getValue(SPState state) {
//            int minDeckSize = Integer.MAX_VALUE;
//            minDeckSize = Math.min(minDeckSize, state.workerDeck.size());
//            minDeckSize = Math.min(minDeckSize, state.buildingDeck.size());
//            minDeckSize = Math.min(minDeckSize, state.aristocratDeck.size());
//            minDeckSize = Math.min(minDeckSize, state.tradingDeck.size());
//            return minDeckSize;
//        }
//    }
//
//    // points – current player points
//    class SPFeaturePoints extends SPFeature {
//        public SPFeaturePoints() {
//            super("points", "current player points");
//        }
//
//        public Object getValue(SPState state) {
//            return state.playerPoints[state.playerTurn];
//        }
//    }
//
//    // points_diff – current player points relative to the opponent (assumes two players)
//    class SPFeaturePointsDiff extends SPFeature {
//        public SPFeaturePointsDiff() {
//            super("points_diff", "current player points relative to the opponent");
//        }
//
//        public Object getValue(SPState state) {
//            return state.playerPoints[state.playerTurn] - state.playerPoints[1 - state.playerTurn];
//        }
//    }
//
//    // rubles – current player rubles (money)
//    class SPFeatureRubles extends SPFeature {
//        public SPFeatureRubles() {
//            super("rubles", "current player rubles (money)");
//        }
//
//        public Object getValue(SPState state) {
//            return state.playerRubles[state.playerTurn];
//        }
//    }
//
//    // rubles_diff – current player rubles (money) relative to the opponent
//    class SPFeatureRublesDiff extends SPFeature {
//        public SPFeatureRublesDiff() {
//            super("rubles_diff", "current player rubles (money) relative to the opponent");
//        }
//
//        public Object getValue(SPState state) {
//            return state.playerRubles[state.playerTurn] - state.playerRubles[1 - state.playerTurn];
//        }
//    }
//
//    // points_round_gain – the number of points the current player is gaining per round
//    class SPFeaturePointsRoundGain extends SPFeature {
//        public SPFeaturePointsRoundGain() {
//            super("points_round_gain", "the number of points the current player is gaining per round");
//        }
//
//        public Object getValue(SPState state) {
//            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
//                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
//                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
//            return pointsPerRound;
//        }
//    }
//
//    // points_round_gain_diff – the number of points the current player is gaining per round relative to the opponent
//    class SPFeaturePointsRoundGainDiff extends SPFeature {
//        public SPFeaturePointsRoundGainDiff() {
//            super("points_round_gain_diff", "the number of points the current player is gaining per round relative to the opponent");
//        }
//
//        public Object getValue(SPState state) {
//            int pointsPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
//                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum()
//                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
//            int opponentPointsPerRound = state.playerWorkers.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
//                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum()
//                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.points).sum();
//            return pointsPerRound - opponentPointsPerRound;
//        }
//    }
//
//    // rubles_round_gain – the number of rubles the current player is gaining per round
//    class SPFeatureRublesRoundGain extends SPFeature {
//        public SPFeatureRublesRoundGain() {
//            super("rubles_round_gain", "the number of rubles the current player is gaining per round");
//        }
//
//        public Object getValue(SPState state) {
//            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
//                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
//                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
//            return rublesPerRound;
//        }
//    }
//
//    // rubles_round_gain_diff – the number of rubles the current player is gaining per round relative to the opponent
//    class SPFeatureRublesRoundGainDiff extends SPFeature {
//        public SPFeatureRublesRoundGainDiff() {
//            super("rubles_round_gain_diff", "the number of rubles the current player is gaining per round relative to the opponent");
//        }
//
//        public Object getValue(SPState state) {
//            int rublesPerRound = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
//                    + state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
//                    + state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
//            int opponentRublesPerRound = state.playerWorkers.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
//                    + state.playerBuildings.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum()
//                    + state.playerAristocrats.get(1 - state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
//            return rublesPerRound - opponentRublesPerRound;
//        }
//    }
//
//    // unique_aristocrats – the number of unique aristocrats of the current player
//    class SPFeatureUniqueAristocrats extends SPFeature {
//        public SPFeatureUniqueAristocrats() {
//            super("unique_aristocrats", "the number of unique aristocrats of the current player");
//        }
//
//        public Object getValue(SPState state) {
//            return state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
//        }
//    }
//
//    // unique_aristocrats_diff – the number of unique aristocrats of the current player relative to the opponent
//    class SPFeatureUniqueAristocratsDiff extends SPFeature {
//        public SPFeatureUniqueAristocratsDiff() {
//            super("unique_aristocrats_diff", "the number of unique aristocrats of the current player relative to the opponent");
//        }
//
//        public Object getValue(SPState state) {
//            long uniqueAristocrats = state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
//            long opponentUniqueAristocrats = state.playerAristocrats.get(1 - state.playerTurn).stream().distinct().count();
//            return uniqueAristocrats - opponentUniqueAristocrats;
//        }
//    }
//
//    // cards_in_hand – the number of cards in the current player hand
//    class SPFeatureCardsInHand extends SPFeature {
//        public SPFeatureCardsInHand() {
//            super("cards_in_hand", "the number of cards in the current player hand");
//        }
//
//        public Object getValue(SPState state) {
//            return state.playerHands.get(state.playerTurn).size();
//        }
//    }
//
//    // cards_in_hand_diff – the number of cards in the current player hand relative to the opponent
//    class SPFeatureCardsInHandDiff extends SPFeature {
//        public SPFeatureCardsInHandDiff() {
//            super("cards_in_hand_diff", "the number of cards in the current player hand relative to the opponent");
//        }
//
//        public Object getValue(SPState state) {
//            int cardsInHand = state.playerHands.get(state.playerTurn).size();
//            int opponentCardsInHand = state.playerHands.get(1 - state.playerTurn).size();
//            return cardsInHand - opponentCardsInHand;
//        }   
//    }
//    
//    // New Class Feature: aristocrats_in_hand_diff
//    class SPFeatureAristocratsInHandsDiff extends SPFeature{
//    	public SPFeatureAristocratsInHandsDiff() {
//    		super("aristocrats_in_hand_diff", "the number of aristocrat cards in the current player hand relative to the opponent");
//    	}
//    	
//    	public Object getValue(SPState state) {
//            int aristocratCardsInHand = 0, opponentAristocratCardsInHand = 0 ;
//            for (SPCard card : state.playerHands.get(state.playerTurn)) {
//            	if (card.isAristocrat) {
//            		aristocratCardsInHand++;
//            	}
//            }
//            
//            for (SPCard card : state.playerHands.get(1-state.playerTurn)) {
//            	if (card.isAristocrat) {
//            		opponentAristocratCardsInHand++;
//            	}
//            }
//            
//            
//            return aristocratCardsInHand - opponentAristocratCardsInHand;
//        }   
//    }
//
//    public static void main(String[] args) {
//        SPStateFeaturesLR2 features = new SPStateFeaturesLR2();
//        features.learnModel();        
//    }
//
//}
//

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import smile.classification.LogisticRegression;

public class MaiNeLR2 {
    String modelFilename = "MaiNeLR2.model";
    LogisticRegression.Binomial model;
    ArrayList<SPFeature> features;
    
    public ArrayList<Object> getFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : features) {
            values.add(feature.getValue(state));
        }
        return values;
    }

    public MaiNeLR2() {
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
//      features.add(new SPFeatureAristocratsInHandsDiff());
//      features.add(new SPFeatureInteractionTerm(new SPFeatureAristocratsInHandsDiff(), new SPFeatureMinDeckSize()));

        initializeModel();
    }

    private void initializeModel() {
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file does not exist. Generating model...");
            learnModel();
        }
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(new java.io.FileInputStream(modelFilename))) {
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
        	System.out.println(row);

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
                
             // SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPRandomPlayer(), new SPRandomPlayer());
           	 // SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPMMustafizPlayer(), new SPMMustafizPlayer());
                SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPPlayerFlatMC(), new SPPlayerFlatMC());
             // SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPCheap(), new SPCheap());

               writer.print(getCSVRows(transcript));
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
        int numGames = 500; // Number of games to simulate for training data
        generateCSVData(trainingDataFile, numGames);

        // Load the training data from the CSV file into a Smile dataset (Anh code)
        List<double[]> values = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
            String line = br.readLine(); 
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

        // Perform logistic regression using the Smile library
        LogisticRegression.Binomial model = LogisticRegression.binomial(X, y);

        // Save the model to a file using an ObjectOutputStream
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(modelFilename))) {
            oos.writeObject(model);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Print the model coefficients along with their feature names
        System.out.println("Model coefficients:");
        System.out.println(features.size() + " features");
        System.out.println(model.coefficients().length + " coefficients");
        System.out.printf("%.4f\tIntercept%n", model.coefficients()[0]);
        for (int i = 0; i < model.coefficients().length - 1; i++) {
            System.out.printf("%.4f\t%s%n", model.coefficients()[i + 1], features.get(i).getName());
        }

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

        // Use the logistic regression model to predict the probability of winning
        return model.score(featureValues);
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
    
 // New Class Feature: aristocrats_in_hand_diff
    class SPFeatureAristocratsInHandsDiff extends SPFeature{
    	public SPFeatureAristocratsInHandsDiff() {
    		super("aristocrats_in_hand_diff", "the number of aristocrat cards in the current player hand relative to the opponent");
    	}
    	
    	public Object getValue(SPState state) {
            int aristocratCardsInHand = 0, opponentAristocratCardsInHand = 0 ;
            for (SPCard card : state.playerHands.get(state.playerTurn)) {
            	if (card.isAristocrat) {
            		aristocratCardsInHand++;
            	}
            }
            
            for (SPCard card : state.playerHands.get(1-state.playerTurn)) {
            	if (card.isAristocrat) {
            		opponentAristocratCardsInHand++;
            	}
            }
            
            
            return aristocratCardsInHand - opponentAristocratCardsInHand;
        }   
    }

    public static void main(String[] args) {
        new MaiNeLR2();
        MaiNeLR2 SP = new MaiNeLR2();
        //SP.learnModel();
//        String a = SP.getCSVHeader();
//        System.out.println(a);
    }

}