import java.util.ArrayList;

public class AIDanSPPlayerHybridObjective extends SPPlayer {

//    int numSimulationsPerAction = 10000;
//    int playoutTerminationDepth = 4;
//    boolean verbose = true;
//    
//    private final SPStateFeaturesLR2 winProbFeatures;
//    private final SPStateFeaturesLinRegScoreDiff scoreDiffFeatures;
//
//    public SPPlayerHybridObjective() {
//        super("SPPlayerHybridObjective");
//        this.winProbFeatures = new SPStateFeaturesLR2();
//        this.scoreDiffFeatures = new SPStateFeaturesLinRegScoreDiff();
//    }
//
//    @Override
//    public int getAction(SPState state) {
//        ArrayList<SPAction> actions = state.getLegalActions();
//        int bestActionIndex = -1;
//        double bestWinProb = Double.NEGATIVE_INFINITY;
//        double bestScoreDiff = Double.NEGATIVE_INFINITY;
//        int numActions = actions.size();
//        if (verbose) System.out.println("Number of legal actions: " + numActions);
//
//        for (int i = 0; i < actions.size(); i++) {
//            double totalWinProb = 0.0;
//            double totalScoreDiff = 0.0;
//
//            for (int j = 0; j < numSimulationsPerAction; j++) {
//                SPState depth1Copy = state.clone();
//                depth1Copy.getLegalActions().get(i).take();
//                
//                SPState simCopy = depth1Copy.clone();
//                
//                for (int k = 0; !simCopy.isGameOver() && k < playoutTerminationDepth; k++) {
//                    ArrayList<SPAction> legalActions = simCopy.getLegalActions();
//                    SPAction randomAction = legalActions.get((int) (Math.random() * legalActions.size()));
//                    randomAction.take();
//                }
//
//                double predictedWinProb = winProbFeatures.predict(simCopy);
//                double predictedScoreDiff = scoreDiffFeatures.predict(simCopy);
//
//                // Adjust values based on whose turn it is in the simulated state.
//                if (state.playerTurn != simCopy.playerTurn) {
//                    predictedWinProb = 1.0 - predictedWinProb;
//                    predictedScoreDiff = -predictedScoreDiff;
//                }
//                
//                totalWinProb += predictedWinProb;
//                totalScoreDiff += predictedScoreDiff;
//            }
//
//            double avgWinProb = totalWinProb / numSimulationsPerAction;
//            double avgScoreDiff = totalScoreDiff / numSimulationsPerAction;
//
//            // CORE LOGIC CHANGE:
//            if (bestActionIndex == -1) { // First action is always the best so far.
//                 bestWinProb = avgWinProb;
//                 bestScoreDiff = avgScoreDiff;
//                 bestActionIndex = i;
//            } else {
//                // Primary Objective: Is the win probability clearly higher?
//                // We use a small epsilon for floating point comparison.
//                if (avgWinProb > bestWinProb + 1e-6) {
//                    bestWinProb = avgWinProb;
//                    bestScoreDiff = avgScoreDiff;
//                    bestActionIndex = i;
//                } 
//                // Secondary (Tie-Breaker) Objective: Is the win probability effectively the same,
//                // AND the score difference is higher?
//                else if (Math.abs(avgWinProb - bestWinProb) < 1e-6 && avgScoreDiff > bestScoreDiff) {
//                    // Note: We only update the tie-breaker value and the index. bestWinProb stays the same.
//                    bestScoreDiff = avgScoreDiff;
//                    bestActionIndex = i;
//                }
//            }
//        }
//
//        if (verbose) {
//            System.out.printf("%s (Best Prob: %.3f, Best Diff: %.2f)\n", 
//                              actions.get(bestActionIndex), bestWinProb, bestScoreDiff);
//        }
//        
//        return bestActionIndex;
//    }
	
	
	int numSimulationsPerAction = 10000;
    int playoutTerminationDepth = 4;
    AIDanSPStateFeaturesLR2 features = new AIDanSPStateFeaturesLR2();
    boolean verbose = false;

    public AIDanSPPlayerHybridObjective() {
      super("AIDanSPPlayerHybridObjective");
  }

    @Override
    public int getAction(SPState state) {
        int bestActionIndex = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        ArrayList<SPAction> actions = state.getLegalActions();
        int numActions = actions.size();
        if (verbose) System.out.println("Number of legal actions: " + numActions);
        for (int i = 0; i < numActions; i++) { // for each legal action
        	// needs to happen again and again for each simulation, because
        	// chance behaviors should be sampled -> these lines belong insideloop
            double estValue = 0.0;
            for (int j = 0; j < numSimulationsPerAction; j++) {
                SPState depth1Copy = state.clone();
                SPAction action = depth1Copy.getLegalActions().get(i);
                action.take();
            	
            	SPState simCopy = depth1Copy.clone();
                
                for (int k = 0; !simCopy.isGameOver() && k < playoutTerminationDepth; k++) {
                    ArrayList<SPAction> legalActions = simCopy.getLegalActions();
                    SPAction randomAction = legalActions.get((int) (Math.random() * legalActions.size()));
                    randomAction.take();
                }
                
                double heuristicValue = eval(simCopy);
                if (state.playerTurn != simCopy.playerTurn) {
                    heuristicValue = 1 - heuristicValue; // assuming two players, the estimated probability of winning is 1 minus the opponent's value
                }
                estValue += heuristicValue;
            }
            estValue /= numSimulationsPerAction;
            if (estValue > bestValue) {
                bestValue = estValue;
                bestActionIndex = i;
            }
        }
        if (verbose) System.out.printf("%s (est. value %.4f)\n", actions.get(bestActionIndex), bestValue);
        return bestActionIndex;
    }

    private double eval(SPState state) {
        return features.predict(state);
    }
}