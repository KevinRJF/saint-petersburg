import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import smile.base.cart.SplitRule;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructType;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.data.vector.ValueVector;
import smile.regression.RandomForest;
import smile.util.IterativeAlgorithmController;

public class MiMaStateFeatures {

	String modelFilename = "MiMaRandomForest1.model";
	RandomForest model;
    ArrayList<SPFeature> features;
    StructType schema;

    public ArrayList<Object> getFeatureValues(SPState state) {
        ArrayList<Object> values = new ArrayList<>();
        for (SPFeature feature : features) {
            values.add(feature.getValue(state));
        }
        return values;
    }

    public MiMaStateFeatures() {
        features = new ArrayList<>();
        features.add(new MiMaFeatureMinDeckSize());
        features.add(new MiMaFeaturePoints());
        features.add(new SPFeatureInteractionTerm(new MiMaFeaturePoints(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeaturePointsDiff());
        features.add(new SPFeatureInteractionTerm(new MiMaFeaturePointsDiff(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureRubles());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureRubles(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureRublesDiff());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureRublesDiff(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeaturePointsRoundGain());
        features.add(new SPFeatureInteractionTerm(new MiMaFeaturePointsRoundGain(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeaturePointsRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new MiMaFeaturePointsRoundGainDiff(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureRublesRoundGain());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureRublesRoundGain(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureRublesRoundGainDiff());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureRublesRoundGainDiff(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureUniqueAristocrats());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureUniqueAristocrats(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureUniqueAristocratsDiff());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureUniqueAristocratsDiff(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureCardsInHand());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureCardsInHand(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureCardsInHandDiff());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureCardsInHandDiff(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureSpotsLeft());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureSpotsLeft(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureOdd());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureOdd(), new MiMaFeatureMinDeckSize()));
        features.add(new MiMaFeatureStartsNext());
        features.add(new SPFeatureInteractionTerm(new MiMaFeatureStartsNext(), new MiMaFeatureMinDeckSize()));
        
        

        initializeModel();
    }

    
    private void initializeModel() {
        if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file does not exist. Generating model...");
            learnModel();
            System.out.println("IF");
        }
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(new java.io.FileInputStream(modelFilename))) {
        	model = (RandomForest) ois.readObject();
        	schema = (StructType) ois.readObject();
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
                SPGameTranscript transcript = SPSimulateGame.simulateGame(new SPRandomPlayer(), new SPRandomPlayer());
                writer.print(getCSVRows(transcript));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void learnModel() {
        // This method assumes that the regression model has not been created and saved yet.
        // It generates training data by simulating games and saves it to a CSV file.
        // Then it uses random forest to learn a model and saves it to a file.

        String trainingDataFile = "SPTrainingData.csv";
        int numGames = 10000; // Number of games to simulate for training data
        generateCSVData(trainingDataFile, numGames);

        // Load the training data from the CSV file into a Smile dataset (Anh code)
        List<double[]> values = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        
        List<String> headers = null;

        try (BufferedReader br = new BufferedReader(new FileReader(trainingDataFile))) {
            String line = br.readLine(); 
            if (line != null) {
            	headers = Arrays.asList(line.split(","));
            }
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

        int n = values.size();
        int p = features.size();
        double[][] X = values.toArray(new double[n][]);
        int[] y = labels.stream().mapToInt(i -> i).toArray();
        
        List<ValueVector> vectors = new ArrayList<>();
        
        for (int j = 0; j < p; j++) {
        	double[] col = getColumn (X, j);
        	String featureName = headers.get(j);
        	DoubleVector dv = new DoubleVector(featureName, col);
        	vectors.add(dv);
        }
        
        String labelName = headers.get(p);
        IntVector iv = new IntVector(labelName, y);
        vectors.add(iv);
        
        DataFrame df = new DataFrame(vectors.toArray(new ValueVector[0]));
        
        Formula formula = Formula.lhs(headers.get(p));
        
        int nTrees = 100;
        int mtry = (int) p/3 ;
        int maxDepth = 6;
        int maxNodes = 0;
        int nodeSize = 1;
        double subsample = 1.0;
        long[] seeds = new long[nTrees];
        for (int i = 0; i < nTrees; i++) {
            seeds[i] = 42 + i;
        }

        IterativeAlgorithmController<RandomForest.TrainingStatus> controller = new IterativeAlgorithmController<RandomForest.TrainingStatus>();
        
        this.schema = df.schema();
     
        
        RandomForest.Options options = new RandomForest.Options(nTrees, mtry, maxDepth, maxNodes, nodeSize, subsample, seeds, controller);

        model = RandomForest.fit(formula, df, options);
        
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilename))) {
        	oos.writeObject(model);
        	oos.writeObject(this.schema);
        }
        catch (IOException e) {
        	e.printStackTrace();
        }
        
        System.out.println("Feature importances: ");
        double[] importance = model.importance();
        for (int i = 1; i < importance.length; i++) {
        	System.out.printf("%s: %.4f%n", features.get(i).getName(), importance[i]);
        }
        
        
        // Delete the training data file after learning the model
        java.nio.file.Path path = java.nio.file.Paths.get(trainingDataFile);
        try {
            java.nio.file.Files.delete(path);
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        
       
    }
    
   private static double[] getColumn (double[][] X, int columnIndex) {
	   double[] col = new double[X.length];
       for (int i = 0; i < X.length; i++) {
       	col[i] = X[i][columnIndex];
       }
       
       return col;
   }
    
    public double predict(SPState state) {
        // Create a double array for the feature values
        double[] featureValues = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object value = features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }
        
        
        Tuple tuple = Tuple.of(schema, featureValues) ;
        return model.predict(tuple);

    }
    

    // min_deck_size – the number of cards in the smallest phase deck
    class MiMaFeatureMinDeckSize extends SPFeature {
        public MiMaFeatureMinDeckSize() {
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
    class MiMaFeaturePoints extends SPFeature {
        public MiMaFeaturePoints() {
            super("points", "current player points");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn];
        }
    }

    // points_diff – current player points relative to the opponent (assumes two players)
    class MiMaFeaturePointsDiff extends SPFeature {
        public MiMaFeaturePointsDiff() {
            super("points_diff", "current player points relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerPoints[state.playerTurn] - state.playerPoints[1 - state.playerTurn];
        }
    }

    // rubles – current player rubles (money)
    class MiMaFeatureRubles extends SPFeature {
        public MiMaFeatureRubles() {
            super("rubles", "current player rubles (money)");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn];
        }
    }

    // rubles_diff – current player rubles (money) relative to the opponent
    class MiMaFeatureRublesDiff extends SPFeature {
        public MiMaFeatureRublesDiff() {
            super("rubles_diff", "current player rubles (money) relative to the opponent");
        }

        public Object getValue(SPState state) {
            return state.playerRubles[state.playerTurn] - state.playerRubles[1 - state.playerTurn];
        }
    }

    // points_round_gain – the number of points the current player is gaining per round
    class MiMaFeaturePointsRoundGain extends SPFeature {
        public MiMaFeaturePointsRoundGain() {
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
    class MiMaFeaturePointsRoundGainDiff extends SPFeature {
        public MiMaFeaturePointsRoundGainDiff() {
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
    class MiMaFeatureRublesRoundGain extends SPFeature {
        public MiMaFeatureRublesRoundGain() {
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
    class MiMaFeatureRublesRoundGainDiff extends SPFeature {
        public MiMaFeatureRublesRoundGainDiff() {
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
    class MiMaFeatureUniqueAristocrats extends SPFeature {
        public MiMaFeatureUniqueAristocrats() {
            super("unique_aristocrats", "the number of unique aristocrats of the current player");
        }

        public Object getValue(SPState state) {
            return state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
        }
    }

    // unique_aristocrats_diff – the number of unique aristocrats of the current player relative to the opponent
    class MiMaFeatureUniqueAristocratsDiff extends SPFeature {
        public MiMaFeatureUniqueAristocratsDiff() {
            super("unique_aristocrats_diff", "the number of unique aristocrats of the current player relative to the opponent");
        }

        public Object getValue(SPState state) {
            long uniqueAristocrats = state.playerAristocrats.get(state.playerTurn).stream().distinct().count();
            long opponentUniqueAristocrats = state.playerAristocrats.get(1 - state.playerTurn).stream().distinct().count();
            return uniqueAristocrats - opponentUniqueAristocrats;
        }
    }

    // cards_in_hand – the number of cards in the current player hand
    class MiMaFeatureCardsInHand extends SPFeature {
        public MiMaFeatureCardsInHand() {
            super("cards_in_hand", "the number of cards in the current player hand");
        }

        public Object getValue(SPState state) {
            return state.playerHands.get(state.playerTurn).size();
        }
    }

    // cards_in_hand_diff – the number of cards in the current player hand relative to the opponent
    class MiMaFeatureCardsInHandDiff extends SPFeature {
        public MiMaFeatureCardsInHandDiff() {
            super("cards_in_hand_diff", "the number of cards in the current player hand relative to the opponent");
        }

        public Object getValue(SPState state) {
            int cardsInHand = state.playerHands.get(state.playerTurn).size();
            int opponentCardsInHand = state.playerHands.get(1 - state.playerTurn).size();
            return cardsInHand - opponentCardsInHand;
        }   
    }
    
    //What I added
    
    //How many spots are left after each phase, tells how many new cards will be drawn in the next phase
    class MiMaFeatureSpotsLeft extends SPFeature {
    	public MiMaFeatureSpotsLeft() {
    		super("spots_left", "how many spots there are for the cards to be shown in the next round");
    	}
    	
    	public Object getValue(SPState state) {
    		ArrayList<SPCard> upperCardRow = state.upperCardRow;
    		int upperRowNum = upperCardRow.size();
    		return 8 - upperRowNum;
    	}
    	
    }
    
    //Whether there are an odd number of spots left for the cards
    class MiMaFeatureOdd extends SPFeature {
    	public MiMaFeatureOdd() {
    		super("odd_or_even", "whether there are an odd number of card spots open");
    	}
    	
    	public Object getValue(SPState state) {
    		ArrayList<SPCard> upperCardRow = state.upperCardRow;
    		int upperRowNum = upperCardRow.size();
    		int spotsLeft = 8 - upperRowNum;
    		if (spotsLeft % 2 != 0) {
    			return 0;
    		}
    		else { 
    			return 1;
    		}
    		
    	}
    }
    
    //Whether the player starts the next round
    class MiMaFeatureStartsNext extends SPFeature {
    	public MiMaFeatureStartsNext() {
    		super("starts_next", "whether the player starts the next round");
    	}
    	
    	public Object getValue(SPState state) {
    		int currPhase = state.phase;
    		int[] startingPlayer = state.startingPlayer;
    		int nextStartingPlayer = startingPlayer[(currPhase + 1) % 4];
    		if (nextStartingPlayer == 1) {
    			return 0;
    		}
    		else {
    			return 1;
    		}
    	}
    }

    public static void main(String[] args) {
        new MiMaStateFeatures();
    }
}

