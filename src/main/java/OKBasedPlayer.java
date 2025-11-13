import java.util.ArrayList;

public class OKBasedPlayer extends SPPlayer {

    private final int numSimulationsPerAction = 1000;
    private final int playoutTerminationDepth = 15;
    private final boolean verbose = true;
    
    // --- THIS IS THE KEY CHANGE ---
    // Use a neural-network-based evaluator implemented in OKBasedFeatures
    OKBasedFeatures features = new OKBasedFeatures();

    public OKBasedPlayer() {
        super("OKBasedPlayer");
    }

    public int getAction(SPState state) {
        int bestActionIndex = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        ArrayList<SPAction> actions = state.getLegalActions();
        int numActions = actions.size();
        
        if (numActions == 0) {
            return -1; // No actions possible
        }

        if (this.verbose) {
            System.out.println("Number of legal actions: " + numActions);
        }

        for(int i = 0; i < numActions; ++i) {
            // Create a copy of the state *after* taking the i-th action
            SPState depth1Copy = state.clone();
            SPAction action = (SPAction)depth1Copy.getLegalActions().get(i);
            action.take();
            
            double estValue = 0.0;

            for(int j = 0; j < this.numSimulationsPerAction; ++j) {
                // Clone the post-action state for each new simulation
                SPState simCopy = depth1Copy.clone();

                for(int k = 0; !simCopy.isGameOver() && k < this.playoutTerminationDepth; ++k) {
                    ArrayList<SPAction> legalActions = simCopy.getLegalActions();
                    if (legalActions.isEmpty()) {
                        break; // Game ended during playout
                    }
                    SPAction randomAction = (SPAction)legalActions.get((int)(Math.random() * (double)legalActions.size()));
                    randomAction.take();
                }

                double heuristicValue = this.eval(simCopy);
                if (state.playerTurn != simCopy.playerTurn) {
                    heuristicValue = 1.0 - heuristicValue;
                }
                estValue += heuristicValue;
            }

            estValue /= (double)this.numSimulationsPerAction;
            
            if (estValue > bestValue) {
                bestValue = estValue;
                bestActionIndex = i;
            }
        }
        
        if (bestActionIndex == -1) {
             bestActionIndex = 0; // Failsafe, should not be hit if numActions > 0
        }

        if (this.verbose) {
            System.out.printf("MyRFPlayer: %s (est. value %.4f)\n", actions.get(bestActionIndex), bestValue);
        }
        return bestActionIndex;
    }

    /**
     * Evaluates the given state using the RandomForest model.
     */
    private double eval(SPState state) {
        return features.predict(state);
    }
}