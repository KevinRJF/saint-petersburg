import java.util.ArrayList;

public class SPPlayerFlatMC extends SPPlayer {

    int numSimulationsPerAction = 10000;
    int playoutTerminationDepth = 4;
    SPStateFeaturesLR1 features = new SPStateFeaturesLR1();
    boolean verbose = false;

    public SPPlayerFlatMC() {
        super("SPPlayerFlatMC");
    }

    @Override
    public int getAction(SPState state) {
        int bestActionIndex = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        ArrayList<SPAction> actions = state.getLegalActions();
        int numActions = actions.size();
        if (verbose) System.out.println("Number of legal actions: " + numActions);
        for (int i = 0; i < numActions; i++) { // for each legal action
//            SPState depth1Copy = state.clone();
//            SPAction action = depth1Copy.getLegalActions().get(i);
//            action.take();
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
