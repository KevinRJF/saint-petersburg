import java.util.ArrayList;
import java.util.Random;

public class JFMTPlayer extends SPPlayer {

    int numSimulationsPerAction = 1000;
    int playoutTerminationDepth = 3;

    JFMTFeatures features = new JFMTFeatures();
    boolean verbose = false;
    Random random = new Random();

    public JFMTPlayer() {
        super("JFMTPlayer");
    }

    @Override
    public int getAction(SPState state) {
        int bestActionIndex = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        ArrayList<SPAction> actions = state.getLegalActions();
        int numActions = actions.size();

        if (verbose) {
            System.out.println("Number of legal actions: " + numActions);
        }
        
        if (numActions == 0) {
            return -1;
        }

        for (int i = 0; i < numActions; i++) {
            double estValue = 0.0;
            
            for (int j = 0; j < numSimulationsPerAction; j++) {
                SPState simState = state.clone();

                ArrayList<SPAction> legalActions = simState.getLegalActions();
                if (i >= legalActions.size()) {
                    continue;
                }
                SPAction action = legalActions.get(i);
                action.take();

                for (int k = 0; !simState.isGameOver() && k < playoutTerminationDepth; k++) {
                    ArrayList<SPAction> playoutActions = simState.getLegalActions();
                    if (playoutActions.isEmpty()) {
                        break;
                    }
                    SPAction randomAction = playoutActions.get(random.nextInt(playoutActions.size()));
                    randomAction.take();
                }

                double heuristicValue = eval(simState);
                
                if (state.playerTurn != simState.playerTurn) {
                    heuristicValue = -heuristicValue;
                }
                estValue += heuristicValue;
            }
            estValue /= numSimulationsPerAction;

            if (estValue > bestValue) {
                bestValue = estValue;
                bestActionIndex = i;
            }
        }
        if (bestActionIndex == -1) {
             if (verbose) {
                System.out.println("No valid actions found during simulation. Returning first action.");
             }
             return 0; 
        }
        if (verbose) {
            System.out.printf("%s (est. value %.4f)\n", actions.get(bestActionIndex), bestValue);
        }
        return bestActionIndex;
    }

    private double eval(SPState state) {
        return features.predict(state);
    }
}

