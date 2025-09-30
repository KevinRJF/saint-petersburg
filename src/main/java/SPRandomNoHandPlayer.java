import java.util.ArrayList;
import java.util.Random;

public class SPRandomNoHandPlayer extends SPPlayer {
	Random random = new Random();

	public SPRandomNoHandPlayer() {
		super("RandomNoHand");
	}

	@Override
	public int getAction(SPState state) {
		ArrayList<SPAction> legalActions = state.getLegalActions();
		ArrayList<Integer> actionIndicesNoHand = new ArrayList<>();
		for (int i = 0; i < legalActions.size(); i++) {
			if (!(legalActions.get(i) instanceof SPAddToHandAction)) {
				actionIndicesNoHand.add(i);
			}
		}
		return actionIndicesNoHand.get(random.nextInt(actionIndicesNoHand.size()));
	}
}