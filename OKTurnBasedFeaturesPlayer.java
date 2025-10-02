import java.util.ArrayList;
import java.util.Random;

public class OKTurnBasedFeaturesPlayer extends SPPlayer {

    private final int numSimulationsPerAction = 200;
    private final int playoutTerminationDepth = 4;
    private final boolean verbose = true;
    OKStateFeaturesLR1 features = new OKStateFeaturesLR1();
    private final Random rng = new Random();
    private final SPFeature roi = features.new ROIFeature();

    public OKTurnBasedFeaturesPlayer() {
        super("OKTurnBasedFeaturesPlayer");
    }

    @Override
    public int getAction(SPState state) {
        ArrayList<SPAction> actions = state.getLegalActions();
        int bestIndex = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        double estValue = 0.0;

        for (int i = 0; i < actions.size(); i++) {
            SPAction curAction = actions.get(i);

            if (curAction instanceof SPBuyAction) {
                // --- SPECIAL: evaluate BUY with ROI ---
                System.out.println("evalbuy");
                estValue = evaluateBuyAction(state, (SPBuyAction) curAction);
            } else{
                for(int j = 0; j < numSimulationsPerAction; j++){
                    SPState simState = state.clone();
                    SPAction simAction = simState.getLegalActions().get(i);
                    simAction.take();

                    SPState simCopy = simState.clone();

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
            }  
            if (estValue > bestValue) {
                bestValue = estValue;
                bestIndex = i;
            }
        }
        if (verbose && bestIndex >= 0) {
            System.out.printf("ROIHybrid chooses %s (value %.3f)%n",
                               actions.get(bestIndex), bestValue);
        }
        return bestIndex >= 0 ? bestIndex : 0;
    }

    //roi eval
    private double evaluateBuyAction(SPState state, SPBuyAction buy) {
        Object roiValue = roi.getValue(state);
        if(roiValue instanceof Number){
            return ((Number) roiValue).doubleValue();
        }
        return 0.0;
    }

    private double eval(SPState state) {
        System.out.println("predict");
        return features.predict(state);
    }

}