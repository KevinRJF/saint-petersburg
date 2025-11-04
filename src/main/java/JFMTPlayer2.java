
import java.util.ArrayList;

public class JFMTPlayer2 extends SPPlayer {

    int numSimulationsPerAction = 1000;
    int playoutTerminationDepth = 4;
    JFMTStateFeaturesNN1 features = new JFMTStateFeaturesNN1();
    boolean verbose = false;//true;
    java.util.Random rng = new java.util.Random();
    double earlyStopThreshold = 0.999; // if an action estimate exceeds this, stop searching

    public JFMTPlayer2() {
        super("JFMTPlayer2");
    }

    @Override
    public int getAction(SPState state) {
        int bestActionIndex = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        ArrayList<SPAction> actions = state.getLegalActions();
        int numActions = actions.size();
        if (verbose)
            System.out.println("Number of legal actions: " + numActions);
        for (int i = 0; i < numActions; i++) { // for each legal action
            // SPState depth1Copy = state.clone();
            // SPAction action = depth1Copy.getLegalActions().get(i);
            // action.take();
            // needs to happen again and again for each simulation, because
            // chance behaviors should be sampled -> these lines belong inside loop
            double estValue = 0.0;
            for (int j = 0; j < numSimulationsPerAction; j++) {
                SPState depth1Copy = state.clone();
                SPAction action = depth1Copy.getLegalActions().get(i);
                action.take();
                SPState simCopy = depth1Copy;
                for (int k = 0; !simCopy.isGameOver() && k < playoutTerminationDepth; k++) {
                    ArrayList<SPAction> legalActions = simCopy.getLegalActions();
                    SPAction randomAction = legalActions.get(rng.nextInt(legalActions.size()));
                    randomAction.take();
                }
                double heuristicValue = eval(simCopy);
                if (state.playerTurn != simCopy.playerTurn) {
                    heuristicValue = 1 - heuristicValue; // the estimated probability of winning is 1 minus the
                                                         // opponent's value
                }
                estValue += heuristicValue;
                if ((estValue / (j + 1)) >= earlyStopThreshold) {
                    // found an action with near-certain win and break out early
                    estValue = estValue / (j + 1);
                    break;
                }
            }
            estValue /= numSimulationsPerAction;
            if (estValue > bestValue) {
                bestValue = estValue;
                bestActionIndex = i;
            }
        }
        if (verbose)
            System.out.printf("%s (est. value %.4f)\n", actions.get(bestActionIndex), bestValue);
        return bestActionIndex;
    }

    private double eval(SPState state) {
        return features.predict(state);
    }
}
