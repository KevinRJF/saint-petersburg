import java.util.ArrayList;
import java.util.Random;

public class SPRandomPlayer extends SPPlayer {
	Random random = new Random();

	public SPRandomPlayer() {
		super("Random");
	}

	@Override
	public int getAction(SPState state) {
		ArrayList<SPAction> legalActions = state.getLegalActions();
		return random.nextInt(legalActions.size());
	}
}