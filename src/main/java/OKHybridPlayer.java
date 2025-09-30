import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import org.apache.commons.csv.CSVFormat;

import smile.classification.LogisticRegression;
import smile.data.formula.Formula;
import smile.regression.LinearModel;
import smile.regression.OLS;

public class OKHybridPlayer extends SPPlayer{
    String modelFilename = "OKScoreDiff.model";


	// Regression model trained on points_diff
	private SPStateFeaturesLR1 features = new SPStateFeaturesLR1();
	int numSimulationsPerAction = 500;
	int playoutTerminationDepth = 4;
	boolean verbose = false;
	LinearModel scoreDiffModel;
	
	public OKHybridPlayer() {
	    super("OKHybridPlayer");
	    loadModels();
	}
	
	private void loadModels() { // modified to avoid redundant model learning - TWN
	    //features.learnModel(); // not necessary - read the constructor code - TWN
	    if (!java.nio.file.Files.exists(java.nio.file.Paths.get(modelFilename))) {
            System.out.println("Model file " + modelFilename + " does not exist. Generating model..."); 
            learnScoreDiffModel();
            // Save the model to a file using an ObjectOutputStream
            try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(modelFilename))) {
                oos.writeObject(scoreDiffModel);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }	
	    try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(new java.io.FileInputStream(modelFilename))) {
	    	scoreDiffModel = (LinearModel) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
	}
	
	private void learnScoreDiffModel() {  //redundant
		String trainingDataFile = "OKSPTrainingScoreData.csv";
		smile.data.DataFrame df = null;
		features.generateCSVData(trainingDataFile, numSimulationsPerAction);
		
		try {
			CSVFormat format = CSVFormat.DEFAULT.builder()
					.setHeader()
					.setSkipHeaderRecord(true)
					.get();
			df = smile.io.Read.csv(trainingDataFile, format);
		}
		catch(FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		catch(IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		catch(URISyntaxException e) {
			e.printStackTrace();
			System.exit(1);
		}
		scoreDiffModel = OLS.fit(Formula.lhs("points_diff"), df);
	}
	
	@Override
	public int getAction(SPState state) {
	    int bestActionIndex = -1;
	    double bestScore = Double.NEGATIVE_INFINITY;
	
	    ArrayList<SPAction> actions = state.getLegalActions();
	    if (verbose) System.out.println("Number of legal actions: " + actions.size());
	
	    for (int i = 0; i < actions.size(); i++) {
	        double totalValue = 0.0;
	
	        for (int j = 0; j < numSimulationsPerAction; j++) {
	            // Copy state and apply action
	            SPState simState = state.clone();
	            SPAction action = simState.getLegalActions().get(i);
	            action.take();
	            SPState simCopy = simState.clone();
	
	            for (int k = 0; !simCopy.isGameOver() && k < playoutTerminationDepth; k++) {
	                ArrayList<SPAction> legal = simCopy.getLegalActions();
	                SPAction randomAction = legal.get((int) (Math.random() * legal.size()));
	                randomAction.take();
	            }
	
	            // Evaluate resulting state with linear regression
	            double heuristicval = evalHybrid(simState, state.playerTurn);
                if (state.playerTurn != simCopy.playerTurn) {
                	heuristicval = 1 - heuristicval; // assuming two players, the estimated probability of winning is 1 minus the opponent's value
                }
                totalValue += heuristicval;
	        }
	
	        totalValue /= numSimulationsPerAction;
	
	        if (totalValue > bestScore) {
	            bestScore = totalValue;
	            bestActionIndex = i;
	        }
	    }
	
        if (verbose) System.out.printf("%s (est. value %.4f)\n", actions.get(bestActionIndex), bestScore);
	    return bestActionIndex;
	}
	
	private double evalHybrid(SPState state, int myIndex) {
		double winProb = features.predict(state);
		double scoreDifference = predictScoreDiff(state, myIndex);
		if(winProb > 0.45 && winProb < 0.55) {
			return scoreDifference;
		}
		else {
			return winProb;
		}
	}
	
	private double predictScoreDiff(SPState state, int myIndex) {
		double scoreDiff = 0.0;
		double[] featureValues = new double[features.features.size()];
        for (int i = 0; i < features.features.size(); i++) {
            Object value = features.features.get(i).getValue(state);
            featureValues[i] = (value instanceof Number) ? ((Number) value).doubleValue() : 0.0;
        }
        scoreDiff = scoreDiffModel.predict(featureValues);
        
        if(state.playerTurn == myIndex) {
        	scoreDiff = -scoreDiff;
        }
        
        return scoreDiff;
	}
}
