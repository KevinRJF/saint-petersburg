import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructType;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.data.vector.ValueVector;
import smile.regression.RandomForest;
import smile.util.IterativeAlgorithmController;

public class MiMaRoundsRemaining {
	
	String modelFilename = "RoundsRemainingModel";
	String dataFilename = "rounds_remaining.csv";
	RandomForest model;
	StructType schema;
	
	public MiMaRoundsRemaining() {
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
	
	public void learnModel() {
		// This method assumes that the regression model has not been created and saved yet.
        // It generates training data by simulating games and saves it to a CSV file.
        // Then it uses random forest to learn a model and saves it to a file.

        String trainingDataFile = dataFilename;

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
        int p = 31;
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
 
       double[] values = new double[31];
       values[0] = state.round;
       values[1] = state.phase;
       values[2] = state.workerDeck.size();
       values[3] = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
       values[4] = state.playerWorkers.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
       values[5] = state.playerWorkers.get((state.playerTurn + 1) % 2).stream().mapToInt(card -> card.points).sum();
       values[6] = state.playerWorkers.get((state.playerTurn + 1) % 2).stream().mapToInt(card -> card.rubles).sum();
       values[7] = state.buildingDeck.size();
       values[8] = state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
       values[9] = state.playerBuildings.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
       values[10] = state.playerBuildings.get((state.playerTurn + 1) % 2).stream().mapToInt(card -> card.points).sum();
       values[11] = state.playerBuildings.get((state.playerTurn + 1) % 2).stream().mapToInt(card -> card.rubles).sum();
       values[12] = state.aristocratDeck.size();
       values[13] = state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.points).sum();
       values[14] = state.playerAristocrats.get(state.playerTurn).stream().mapToInt(card -> card.rubles).sum();
       values[15] = state.playerAristocrats.get((state.playerTurn + 1) % 2).stream().mapToInt(card -> card.points).sum();
       values[16] = state.playerAristocrats.get((state.playerTurn + 1) % 2).stream().mapToInt(card -> card.rubles).sum();
       values[17] = state.tradingDeck.size();
       int minDeckSize = Integer.MAX_VALUE;
       minDeckSize = Math.min(minDeckSize, state.workerDeck.size());
       minDeckSize = Math.min(minDeckSize, state.buildingDeck.size());
       minDeckSize = Math.min(minDeckSize, state.aristocratDeck.size());
       minDeckSize = Math.min(minDeckSize, state.tradingDeck.size());
       values[18] = minDeckSize;
       values[19] = state.playerPoints[0];
       values[20] = state.playerRubles[0];
       int uniqueAristocrats = state.playerAristocrats.get(0).stream().distinct().mapToInt(card -> card.points).sum();
       values[21] = (uniqueAristocrats * (uniqueAristocrats + 1) / 2);
       Set<SPCard> uniqueAristocrats2 = new HashSet<>(state.playerAristocrats.get(0));
       values[22] = state.playerAristocrats.get(0).size() - uniqueAristocrats2.size();
       values[23] = state.playerHands.get(0).size();
       values[24] = state.playerPoints[1];
       values[25] = state.playerRubles[1];
       int uniqueAristocrats3 = state.playerAristocrats.get(1).stream().distinct().mapToInt(card -> card.points).sum();
       values[26] = (uniqueAristocrats3 * (uniqueAristocrats3 + 1) / 2);
       Set<SPCard> uniqueAristocrats4 = new HashSet<>(state.playerAristocrats.get(1));
       values[27] = state.playerAristocrats.get(1).size() - uniqueAristocrats4.size();
       values[28] = state.playerHands.get(1).size();
       values[29] = state.upperCardRow.size();
       values[30] = state.lowerCardRow.size();
        
        
        Tuple tuple = Tuple.of(schema, values) ;

        return model.predict(tuple);
    }
	
}
